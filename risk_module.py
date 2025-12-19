# risk_module.py
# model="doubao-seed-1-6-251015"
# risk_module.py
import os
from dataclasses import dataclass, asdict
from html import escape
from typing import List, Dict

import gradio as gr
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from volcenginesdkarkruntime import Ark

from storage import save_risk_session

import numpy as np
from fer import FER

# =========================
# 0. 豆包 Ark 客户端
# =========================

ARK_API_KEY = os.environ.get("ARK_API_KEY")
ark_client = None
if ARK_API_KEY:
    try:
        ark_client = Ark(api_key=ARK_API_KEY)
        print("[risk_module] Doubao Ark client 初始化成功")
    except Exception as e:
        print("[risk_module] Doubao Ark client 初始化失败:", e)
        ark_client = None
else:
    print("[risk_module] 未检测到 ARK_API_KEY 环境变量，将使用固定模版问题。")

# =========================
# 0.1 表情识别器（FER）
# =========================

face_detector = FER(mtcnn=True)

EMO_CN_MAP = {
    "angry": "愤怒",
    "disgust": "厌恶",
    "fear": "害怕",
    "happy": "高兴",
    "sad": "伤心",
    "surprise": "惊讶",
    "neutral": "中性",
}

NEG_EMOS = {"angry", "disgust", "fear", "sad"}


def analyze_face_emotion(image: np.ndarray):
    """对单帧图像做一次表情识别"""
    if image is None:
        return None
    try:
        results = face_detector.detect_emotions(image)
        if not results:
            return None
        emotions = results[0]["emotions"]  # dict emotion -> prob
        emo, score = max(emotions.items(), key=lambda kv: kv[1])
        return {
            "emotion": emo,
            "score": float(score),
            "emotions": {k: float(v) for k, v in emotions.items()},
        }
    except Exception as e:
        print("[risk_module] analyze_face_emotion error:", e)
        return None


# =========================
# 1. 加载中文情感模型（RoBERTa）
# =========================

MODEL_NAME = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
sentiment_model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
sentiment_model.eval()

# =========================
# 2. 访谈问题 & 高危关键词
# =========================

QUESTIONS = [
    "最近一段时间，你总体的心情怎么样？可以用自己的话简单说说吗？",
    "最近你的睡眠情况如何？比如入睡难不难、中途会不会醒来、总睡多久？",
    "对以前感兴趣的事情（游戏、社交、爱好等），兴趣有没有明显下降？可以举个例子。",
    "你最近是不是经常感到疲惫、没力气、提不起精神？大概是什么时候开始的？",
    "你会不会经常出现自责、觉得自己一无是处或者什么都做不好这样的想法？可以具体描述一下。",
    "最近是否经常感到紧张、焦虑、担心很多事控制不住？如果有，通常在担心些什么？",
    "在学习、工作或人际关系方面，最近有什么让你特别困扰的事情吗？",
    "有没有出现过“活着没意义”“不如消失算了”之类的想法？如果有，出现得有多频繁？",
    "最后，你还有什么想补充的情况，觉得对了解你现在的状态很重要的吗？"
]
TOTAL_QUESTIONS = len(QUESTIONS)

HIGH_RISK_KEYWORDS = [
    "想死", "自杀", "结束生命", "不想活", "活着没意义",
    "消失", "了断", "活得好累", "受不了", "撑不住"
]

# =========================
# 3. 文本 + 表情 → 风险评分
# =========================

def sentiment_negative_prob(text: str) -> float:
    if not text.strip():
        return 0.0

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = sentiment_model(**encoded)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        neg_prob = float(probs[0].item())
    return neg_prob


def compute_face_stats(face_emotions: List[dict]):
    """
    统计人脸表情：
    - counts: 各表情出现次数
    - neg_ratio: 负向表情占比 (0~1)；没有数据时为 None
    """
    if not face_emotions:
        return {}, None

    counts: Dict[str, int] = {}
    for emo in face_emotions:
        label = emo.get("emotion")
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return counts, None

    neg_count = sum(counts.get(e, 0) for e in NEG_EMOS)
    neg_ratio = neg_count / float(total)
    return counts, neg_ratio


def compute_risk_score(all_answers_text: str, face_emotions: List[dict] | None = None) -> Dict:
    """
    综合风险评分：
    - 文本负向概率 → 0~6 分
    - 高危关键词 → 0~2 分
    - 表情负向比例 → 0~2 分
    总分控制在 0~10
    """
    text = all_answers_text or ""
    neg_prob = sentiment_negative_prob(text)  # 0~1

    # 文本主分：0~6
    base_score = neg_prob * 6.0

    # 关键词附加：0~2
    hit_words = [kw for kw in HIGH_RISK_KEYWORDS if kw in text]
    kw_extra = 2.0 if hit_words else 0.0

    # 表情：0~2
    face_counts, face_neg_ratio = compute_face_stats(face_emotions or [])
    if face_neg_ratio is None:
        face_extra = 0.0
    else:
        face_extra = face_neg_ratio * 2.0  # 全程几乎都是负向时，给满 2 分

    score = max(0.0, min(10.0, base_score + kw_extra + face_extra))

    if score >= 7.0:
        level = "高风险（文本与表情均显示较强负向倾向，建议尽快联系专业心理医生或精神科评估）"
    elif score >= 4.0:
        level = "中等风险（存在较明显的负向情绪，建议尽快预约心理咨询或门诊，进一步评估）"
    else:
        level = "低风险（当前整体负向程度偏低，但如不适持续或加重，仍建议及时求助）"

    return {
        "score": round(score, 2),
        "level": level,
        "neg_prob": round(neg_prob, 3),
        "hit_words": list(set(hit_words)),
        "face_neg_ratio": None if face_neg_ratio is None else round(face_neg_ratio, 3),
        "face_counts": face_counts,
    }

# =========================
# 4. 豆包 LLM：共情 + 下一题
# =========================

def llm_empathetic_reply(prev_question: str, user_message: str, next_question: str) -> str:
    if ark_client is None:
        return f"（当前未连接大模型，暂时用固定问题继续）\n好的，感谢你的回答。\n\n{next_question}"

    prompt = (
        "你是一个中文心理支持助手，只做情绪支持和引导提问，不做诊断，不提具体药物。\n\n"
        "刚才你问对方的问题是：\n"
        f"{prev_question}\n\n"
        "来访者刚刚的回答是：\n"
        f"{user_message}\n\n"
        "现在请你做两件事：\n"
        "1. 先用 1~2 句自然、真诚的话回应和共情对方的感受。\n"
        "   要求：尽量引用对方回答里的关键词，并结合你刚才提问的内容来理解这句话。\n"
        "2. 换一行，紧接着问下面这个问题，引导对方继续回答：\n"
        f"{next_question}\n\n"
        "整体要求：\n"
        "- 总字数控制在 80~150 字以内。\n"
        "- 不要自称医生，不要给出任何医学诊断或具体治疗方案。\n"
        "- 不要用特别官方、教科书式的语气，保持口语化、真诚一点。\n"
    )

    try:
        completion = ark_client.chat.completions.create(
            # ⚠ 换成你自己的豆包模型/接入点 ID
            model="doubao-seed-1-6-251015",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个谨慎、温和的中文心理陪伴助手，只做情绪支持和提问，不做诊断。"
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        reply = completion.choices[0].message.content
        return reply.strip()
    except Exception as e:
        print("[risk_module] Doubao error:", e)
        return f"（大模型调用出错，已切换为固定模版）\n好的，感谢你的回答。\n\n{next_question}"

# =========================
# 5. 对话状态 & 逻辑
# =========================

@dataclass
class DialogState:
    step: int = 0
    answers: List[str] = None
    finished: bool = False
    saved: bool = False
    face_emotions: List[dict] = None    # 自动采样到的表情结果
    face_frame_counter: int = 0         # 用于“每隔几帧采一次样”

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        if d is None:
            return DialogState(
                step=0, answers=[], finished=False,
                saved=False, face_emotions=[], face_frame_counter=0
            )
        return DialogState(
            step=d.get("step", 0),
            answers=d.get("answers") or [],
            finished=d.get("finished", False),
            saved=d.get("saved", False),
            face_emotions=d.get("face_emotions") or [],
            face_frame_counter=d.get("face_frame_counter", 0),
        )


def chat_logic(user_message: str, state_dict: dict):
    state = DialogState.from_dict(state_dict)

    # 初次进入
    if state.step == 0 and not state.finished:
        bot_reply = (
            "你好，我是一个心理健康风险评估助手。\n\n"
            "接下来我会问你几组关于情绪、睡眠和压力的简单问题，你可以按真实情况回答。"
            "这些内容只用于研究和自助评估，不能替代正式的心理/精神科诊断。\n\n"
            f"第 1 个问题：{QUESTIONS[0]}"
        )
        state.step = 1
        return bot_reply, state.to_dict()

    # 已完成
    if state.finished:
        bot_reply = (
            "本轮风险评估已经结束。\n\n"
            "如果你的状态有明显变化，可以点击右下角“重新开始”再做一次评估。\n"
            "如果此刻有非常强烈的痛苦感受或安全风险，请尽快联系身边可信任的人或专业机构。"
        )
        return bot_reply, state.to_dict()

    # 访谈中：记录当前回答
    if state.step > 0 and state.step <= TOTAL_QUESTIONS:
        state.answers.append((user_message or "").strip())

    # 还没问完 → 下一题
    if state.step < TOTAL_QUESTIONS:
        prev_q = ""
        if state.step - 1 >= 0:
            prev_q = QUESTIONS[state.step - 1]

        raw_q = QUESTIONS[state.step]
        next_q_full = f"第 {state.step + 1} 个问题：{raw_q}"

        bot_reply = llm_empathetic_reply(prev_q, user_message or "", next_q_full)

        state.step += 1
        return bot_reply, state.to_dict()

    # 问完所有题 → 做风险评估（文本 + 表情）
    if state.step >= TOTAL_QUESTIONS and not state.finished:
        all_text = "\n".join(state.answers)
        risk = compute_risk_score(all_text, state.face_emotions)

        summary = (
            "感谢你认真完成了这些问题。\n\n"
            "【非正式、仅供参考的综合风险评估结果】\n"
            f"- 综合风险得分（0~10）：{risk['score']}\n"
            f"- 文本负向情绪概率：{risk['neg_prob']}\n"
            f"- 风险等级：{risk['level']}\n"
        )
        if risk["hit_words"]:
            summary += f"- 在你的描述中出现了部分高关注语句：{', '.join(risk['hit_words'])}\n"

        # 表情部分说明
        if risk["face_neg_ratio"] is not None:
            summary += (
                f"- 表情负向占比（愤怒/厌恶/害怕/伤心）：{risk['face_neg_ratio']} "
                "(0 表示几乎没有负向表情，1 表示大部分时间是负向表情)\n"
            )
        face_counts = risk["face_counts"]
        if face_counts:
            summary += "【表情采样统计】\n"
            total = sum(face_counts.values())
            for label, cnt in face_counts.items():
                cn = EMO_CN_MAP.get(label, label)
                summary += f"- {cn}：{cnt} 次（约占 {round(cnt / total * 100)}%）\n"
            summary += "（表情识别仅基于摄像头瞬时画面，可能有误差，仅作辅助参考。）\n"

        summary += (
            "\n请注意：\n"
            "1. 这个结果只是基于文本和表情模型的粗略评估，不能用于临床诊断。\n"
            "2. 如果你已经在生活中感到明显的痛苦、功能受损，或有自伤/自杀的冲动，"
            "请务必尽快联系学校心理中心、医院精神科或当地心理援助热线。\n"
            "3. 建议你把线下的专业求助放在更重要的位置，这个系统更多用于自助筛查与提醒。"
        )

        state.finished = True
        return summary, state.to_dict()

    # 兜底
    bot_reply = "系统状态有点异常，请尝试刷新页面或点击“重新开始”重新评估。"
    return bot_reply, state.to_dict()

# =========================
# 6. UI：聊天 HTML（气泡）
# =========================

def append_message_html(inner_html: str, role: str, text: str) -> str:
    if not text:
        return inner_html

    safe = escape(text).replace("\n", "<br>")

    if role == "user":
        bubble = f"""
        <div class="chat-row chat-row-user">
          <div class="chat-bubble chat-bubble-user">{safe}</div>
        </div>
        """
    else:
        bubble = f"""
        <div class="chat-row chat-row-bot">
          <div class="chat-bubble chat-bubble-bot">{safe}</div>
        </div>
        """

    return inner_html + bubble


def build_chat_card_html(inner_html: str) -> str:
    return f"""
    <div class="card chat-card">
      <div class="chat-scroll">
        {inner_html}
      </div>
    </div>
    """

# =========================
# 7. 摄像头流事件：自动采集表情
# =========================

def on_face_stream(frame, state_dict):
    """
    每一帧摄像头图像都会触发这个函数（stream 模式）。
    为了不太吃算力，只每 N 帧做一次表情识别。
    """
    state = DialogState.from_dict(state_dict)
    state.face_frame_counter += 1

    # 这里假设摄像头 ~10fps，每 20 帧 ≈ 2 秒采一次；你可以自己调 N
    N = 20
    if state.face_frame_counter % N == 0:
        emo = analyze_face_emotion(frame)
        if emo:
            if state.face_emotions is None:
                state.face_emotions = []
            state.face_emotions.append(emo)

    return state.to_dict()

# =========================
# 8. 主对话响应（带持久化）
# =========================

def respond_with_persist(user_message, inner_html, state_dict, email):
    """
    聊天逻辑 + 回车发送 + 写入数据库（按 email）
    摄像头采样已通过 on_face_stream 不断更新 state.face_emotions
    """
    new_inner = inner_html
    if user_message:
        new_inner = append_message_html(new_inner, "user", user_message)

    bot_reply, new_state = chat_logic(user_message, state_dict)
    new_inner = append_message_html(new_inner, "bot", bot_reply)

    # 持久化：只在第一次 finished 且 email 不为空时保存
    if new_state.get("finished") and not new_state.get("saved"):
        email = (email or "").strip().lower()
        if email:
            answers = new_state.get("answers") or []
            face_emotions = new_state.get("face_emotions") or []
            try:
                risk = compute_risk_score("\n".join(answers), face_emotions)
                save_risk_session(email, risk, answers)
                new_state["saved"] = True
                print(f"[risk_module] 已保存一条评估记录给 {email}")
            except Exception as e:
                print("[risk_module] save_risk_session error:", e)

    outer_html = build_chat_card_html(new_inner)
    return "", outer_html, new_inner, new_state


def clear_all():
    empty_inner = ""
    outer_html = build_chat_card_html(empty_inner)
    new_state = DialogState(
        step=0, answers=[], finished=False,
        saved=False, face_emotions=[], face_frame_counter=0
    ).to_dict()
    return outer_html, empty_inner, new_state

# =========================
# 9. 对外入口：build_risk_page
# =========================

def build_risk_page(user_email_state: gr.State):
    """
    在已有 Blocks 上构建“文本 + 表情 风险评估助手”页面。
    """
    with gr.Column(visible=False) as risk_page:
        gr.HTML(
            """
            <div class="app-header">
              <div class="app-title">
                <span class="icon">🧠</span>
                <span>文本 + 表情 风险评估助手</span>
              </div>
              <div class="app-subtitle">
                实时采集摄像头表情 + 文本访谈，对当前情绪状态进行非正式的风险评估与提醒。
              </div>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=4):
                gr.HTML(
                    """
                    <div class="card">
                      <div class="side-title">使用说明</div>
                      <div class="side-text">
                        · 允许浏览器访问摄像头，保持脸部尽量在画面中。<br>
                        · 系统会每隔几秒自动采集一次表情，不需要手动拍照。<br>
                        · 同时按提示逐个回答右侧对话中的问题。
                      </div>

                      <div class="side-title" style="margin-top:10px;">风险等级</div>
                      <div class="side-text">
                        <span class="side-tag">0-3 分 · 低风险</span>
                        <span class="side-tag">4-6 分 · 中等风险</span>
                        <span class="side-tag">7-10 分 · 高风险</span><br>
                        分数越高，只代表文本和表情中负向倾向越明显，<strong>不等同于临床诊断</strong>。
                      </div>

                      <div class="side-title" style="margin-top:10px;">重要提醒</div>
                      <div class="side-text">
                        如果你已经出现持续失眠、明显功能受损，或有自伤/自杀想法，<br>
                        请立刻联系学校心理中心、医院精神科或当地心理援助热线，<br>
                        不要仅依赖本系统做重要决策。
                      </div>
                    </div>
                    """
                )

            with gr.Column(scale=6):
                state = gr.State(
                    DialogState(
                        step=0, answers=[], finished=False,
                        saved=False, face_emotions=[], face_frame_counter=0
                    ).to_dict()
                )
                chat_inner_state = gr.State("")

                history_html = gr.HTML(value=build_chat_card_html(""))

                # 摄像头实时流（不用点拍照）
                face_feed = gr.Image(
                    label="摄像头（系统会自动采集表情）",
                    sources=["webcam"],
                    streaming=True,
                    type="numpy",
                )

                msg = gr.Textbox(
                    show_label=False,
                    placeholder="在这里输入你的回答，然后按回车或点击右侧按钮发送。",
                    lines=1,
                )

        with gr.Row():
            send_btn = gr.Button("发送 / 下一题")
            clear_btn = gr.Button("重新开始")
            back_btn = gr.Button("⬅ 返回主页面")

    # 摄像头流 → 自动采样表情，更新 state
    face_feed.stream(
        on_face_stream,
        inputs=[face_feed, state],
        outputs=[state],
    )

    # 聊天逻辑
    msg.submit(
        respond_with_persist,
        inputs=[msg, chat_inner_state, state, user_email_state],
        outputs=[msg, history_html, chat_inner_state, state],
    )
    send_btn.click(
        respond_with_persist,
        inputs=[msg, chat_inner_state, state, user_email_state],
        outputs=[msg, history_html, chat_inner_state, state],
    )
    clear_btn.click(
        clear_all,
        inputs=[],
        outputs=[history_html, chat_inner_state, state],
    )

    controls = {
        "state": state,
        "chat_inner_state": chat_inner_state,
        "history_html": history_html,
        "msg": msg,
        "send_btn": send_btn,
        "clear_btn": clear_btn,
        "back_btn": back_btn,
        "face_feed": face_feed,
    }
    return risk_page, controls

