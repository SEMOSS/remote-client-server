from globals.app_instance import get_app


def get_server_status():
    app = get_app()
    return app.state.server_status


def set_server_status(status: str):
    app = get_app()
    app.state.server_status = status
