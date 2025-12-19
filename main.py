#激活虚拟环境； .\.venv\Scripts\Activate.ps1
#如果要运行live server，需要开启vpn的Tunnel模式，否则会timeout
import gradio as gr

from risk_module import build_risk_page, QUESTIONS
from storage import list_risk_sessions, get_risk_session, register_or_check_user




def do_login(email_input: str, password_input: str):
    email = (email_input or "").strip().lower()
    if not email or "@" not in email:
        return (
            "❌ 邮箱格式不太对，请重新输入。",
            "",
            False,
            gr.update(visible=True),   # login_page
            gr.update(visible=False),  # home_page
            gr.update(visible=False),  # history_page
            gr.update(visible=False),  # risk_page
        )

    password = (password_input or "").strip()
    if not password:
        return (
            "❌ 请输入密码。",
            "",
            False,
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    created, ok = register_or_check_user(email, password)
    if not ok:
        return (
            "❌ 密码错误，请重试。如果你忘记密码，只能让管理员手动重置数据库中的该账号。",
            "",
            False,
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    if created:
        msg = f"✅ 已为 {email} 创建新账号，请记住当前密码。"
    else:
        msg = f"✅ 登录成功：{email}"

    return (
        msg,
        email,
        True,
        gr.update(visible=False),   # login_page hidden
        gr.update(visible=True),    # home_page visible
        gr.update(visible=False),   # history_page hidden
        gr.update(visible=False),   # risk_page hidden
    )


def refresh_history(email: str):
    email = (email or "").strip().lower()
    if not email:
        return [], gr.update(choices=[], value=None), "⚠ 请先登录。"

    records = list_risk_sessions(email)
    if not records:
        return [], gr.update(choices=[], value=None), "当前没有任何历史记录，可以先做一次评估。"

    table_data = []
    choices = []
    for r in records:
        table_data.append([r["id"], r["ts"], r["score"], r["level"]])
        choices.append(str(r["id"]))

    return table_data, gr.update(choices=choices, value=None), f"共找到 {len(records)} 条评估记录。"


def show_session_detail(session_id_str: str, email: str):
    email = (email or "").strip().lower()
    if not email:
        return "⚠ 请先登录。"

    if not session_id_str:
        return "请先在上方选择一条记录。"

    try:
        sid = int(session_id_str)
    except ValueError:
        return "选择的记录 ID 非法。"

    detail = get_risk_session(email, sid)
    if detail is None:
        return "未找到这条记录，或该记录不属于当前登录账号。"

    md_lines = []
    md_lines.append(f"### 记录 ID：{detail['id']}")
    md_lines.append(f"- 时间：{detail['ts']}")
    md_lines.append(f"- 综合风险得分：{detail['score']}")
    md_lines.append(f"- 风险等级：{detail['level']}")
    md_lines.append(f"- 文本负向情绪概率：{detail['neg_prob']}")
    hit_words = detail.get("hit_words") or []
    if hit_words:
        md_lines.append(f"- 高关注词语：{', '.join(hit_words)}")
    md_lines.append("")
    md_lines.append("#### 问题与原始回答")
    answers = detail.get("answers") or []
    for idx, ans in enumerate(answers, start=1):
        q = QUESTIONS[idx - 1] if idx - 1 < len(QUESTIONS) else f"第 {idx} 题"
        md_lines.append(f"**Q{idx}：{q}**")
        md_lines.append("")
        md_lines.append(f"{ans or '（无回答）'}")
        md_lines.append("")

    return "\n".join(md_lines)


def logout():
    return (
        "已退出登录。",
        "",
        False,
        gr.update(visible=True),   # login_page
        gr.update(visible=False),  # home_page
        gr.update(visible=False),  # history_page
        gr.update(visible=False),  # risk_page
    )


with gr.Blocks(css="""
:root {
  --bg: #0f172a;
  --card-bg: #111827;
  --accent: #38bdf8;
  --accent-soft: rgba(56,189,248,0.14);
  --border-subtle: rgba(148,163,184,0.35);
  --text-main: #e5e7eb;
  --text-soft: #94a3b8;
}

body { background: var(--bg); }

.app-header {
  margin-bottom: 12px;
}
.app-title {
  font-size: 24px;
  font-weight: 600;
  color: var(--text-main);
  display: flex;
  align-items: center;
  gap: 8px;
}
.app-title .icon {
  font-size: 26px;
}
.app-subtitle {
  margin-top: 6px;
  font-size: 14px;
  color: var(--text-soft);
}
.card {
  background: radial-gradient(circle at top left, rgba(56,189,248,0.12), transparent 55%);
  background-color: var(--card-bg);
  border-radius: 14px;
  border: 1px solid var(--border-subtle);
  padding: 14px 16px;
  box-shadow: 0 18px 45px rgba(15,23,42,0.65);
  margin-top: 10px;
}
.side-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--text-main);
  margin-bottom: 4px;
}
.side-text {
  font-size: 13px;
  color: var(--text-soft);
  line-height: 1.5;
}
.side-tag {
  display: inline-block;
  padding: 2px 8px;
  margin: 2px 6px 2px 0;
  border-radius: 999px;
  border: 1px solid rgba(148,163,184,0.5);
  background: rgba(15,23,42,0.7);
  font-size: 11px;
  color: var(--text-main);
}
""") as demo:

    user_email_state = gr.State("")
    logged_in_state = gr.State(False)

    # ========== 登录页 ==========
    with gr.Column(visible=True) as login_page:
        gr.HTML(
            """
            <div class="app-header">
              <div class="app-title">
                <span class="icon">🔐</span>
                <span>登录 · 心理智能体</span>
              </div>
              <div class="app-subtitle">
                使用邮箱 + 密码登录。首次使用某个邮箱时会自动创建账号，之后需要输入相同密码才能访问该邮箱下的数据。
              </div>
            </div>
            """
        )

        email_box = gr.Textbox(
            label="邮箱",
            placeholder="例如：example@hust.edu.cn",
        )
        password_box = gr.Textbox(
            label="密码",
            type="password",
            placeholder="至少 6 位，首次会自动注册该邮箱账号",
        )

        login_btn = gr.Button("登录", variant="primary")
        login_msg = gr.Markdown()

        gr.HTML(
            """
            <div class="card">
              <div class="side-title">说明</div>
              <div class="side-text">
                · 首次用某个邮箱 + 密码登录时，会在本地创建账号。<br>
                · 之后再次登录，同一邮箱必须用相同密码，否则无法访问该账号下的历史评估记录。<br>
                · 所有数据保存在本地 SQLite 数据库文件中，如果别人拿到你的电脑或数据库文件，理论上仍能直接读取。<br>
                · 正式对外部署时，还需要配合服务器端认证、加密存储等安全措施。
              </div>
            </div>
            """
        )

    # ========== 主页面 ==========
    with gr.Column(visible=False) as home_page:
        gr.HTML(
            """
            <div class="app-header">
              <div class="app-title">
                <span class="icon">🏠</span>
                <span>心理辅助智能体 · 功能中心</span>
              </div>
              <div class="app-subtitle">
                请选择你要使用的模块。目前已开放：文本 + 表情风险评估；后续可以继续扩展其它功能。
              </div>
            </div>
            """
        )

        current_user = gr.Markdown(value="当前用户：未登录")

        go_risk_btn = gr.Button("🧠 文本 + 表情风险评估助手")
        go_history_btn = gr.Button("📊 查看我的历史评估记录")
        logout_btn = gr.Button("🚪 退出登录")

    # ========== 历史记录页面 ==========
    with gr.Column(visible=False) as history_page:
        gr.HTML(
            """
            <div class="app-header">
              <div class="app-title">
                <span class="icon">📊</span>
                <span>我的历史评估记录</span>
              </div>
              <div class="app-subtitle">
                仅展示当前登录账号下的评估记录。你可以选择其中一条查看详细回答与评分情况。
              </div>
            </div>
            """
        )

        history_info = gr.Markdown(value="点击下方的“刷新我的历史记录”获取最新数据。")

        refresh_btn = gr.Button("🔄 刷新我的历史记录")

        history_table = gr.Dataframe(
            headers=["记录ID", "时间", "风险得分", "风险等级"],
            datatype=["str", "str", "str", "str"],
            row_count=(0, "dynamic"),
            column_count=(4, "fixed"),
            interactive=False,
            label="评估记录列表",
        )

        session_dropdown = gr.Dropdown(
            label="选择记录 ID 查看详情",
            choices=[],
        )

        detail_md = gr.Markdown(label="详细内容")
        back_from_history_btn = gr.Button("⬅ 返回主页面")

    # ========== 风险评估页面（来自 risk_module） ==========
    risk_page, risk_controls = build_risk_page(user_email_state)

    # ===== 事件绑定 =====

    # 登录按钮
    login_btn.click(
        do_login,
        inputs=[email_box, password_box],
        outputs=[
            login_msg,         # 文本提示
            user_email_state,  # 当前邮箱
            logged_in_state,   # 是否登录
            login_page,        # login_page.visible
            home_page,         # home_page.visible
            history_page,      # history_page.visible
            risk_page,         # risk_page.visible
        ],
    )

    # 主页显示当前用户
    def update_home_user(email: str):
        email = (email or "").strip().lower()
        if not email:
            return "当前用户：未登录"
        return f"当前用户：**{email}**"

    # 从主页进入风险评估
    go_risk_btn.click(
        lambda email: (
            gr.update(visible=False),  # login
            gr.update(visible=False),  # home
            gr.update(visible=False),  # history
            gr.update(visible=True),   # risk
            update_home_user(email),
        ),
        inputs=[user_email_state],
        outputs=[login_page, home_page, history_page, risk_page, current_user],
    )

    # 从主页进入历史记录
    go_history_btn.click(
        lambda email: (
            gr.update(visible=False),  # login
            gr.update(visible=False),  # home
            gr.update(visible=True),   # history
            gr.update(visible=False),  # risk
            update_home_user(email),
        ),
        inputs=[user_email_state],
        outputs=[login_page, home_page, history_page, risk_page, current_user],
    )

    # 退出登录
    logout_btn.click(
        logout,
        inputs=[],
        outputs=[
            login_msg,
            user_email_state,
            logged_in_state,
            login_page,
            home_page,
            history_page,
            risk_page,
        ],
    )

    # 历史页：刷新列表
    refresh_btn.click(
        refresh_history,
        inputs=[user_email_state],
        outputs=[history_table, session_dropdown, history_info],
    )

    # 历史页：查看详情
    session_dropdown.change(
        show_session_detail,
        inputs=[session_dropdown, user_email_state],
        outputs=[detail_md],
    )

    # 历史页返回主页
    back_from_history_btn.click(
        lambda: (
            gr.update(visible=False),  # login
            gr.update(visible=True),   # home
            gr.update(visible=False),  # history
            gr.update(visible=False),  # risk
        ),
        inputs=[],
        outputs=[login_page, home_page, history_page, risk_page],
    )

    # 风险页里的“返回主页面”按钮
    risk_controls["back_btn"].click(
        lambda: (
            gr.update(visible=False),  # login
            gr.update(visible=True),   # home
            gr.update(visible=False),  # history
            gr.update(visible=False),  # risk
        ),
        inputs=[],
        outputs=[login_page, home_page, history_page, risk_page],
    )

demo.queue().launch(
    share=True,          # 开公网隧道
    server_name="0.0.0.0",
    server_port=7860,
)

