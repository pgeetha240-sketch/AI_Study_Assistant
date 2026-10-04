import threading
import time
import webview

from main import app


def run_flask():
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )


if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    time.sleep(2)

    webview.create_window(
        "AI Study Assistant",
        "http://127.0.0.1:5000",
        width=1000,
        height=750,
        min_size=(800, 600)
    )

    webview.start()