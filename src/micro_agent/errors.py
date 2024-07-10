class AgentExecToolError(Exception):
    def __init__(self, message):
        self.message = message


class RegisterToolError(Exception):
    def __init__(self, message):
        self.message = message
