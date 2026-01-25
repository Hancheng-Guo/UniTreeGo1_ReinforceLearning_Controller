from stable_baselines3.common.logger import Logger

class LoggerProxy:
    def __init__(self, logger: Logger, prefix_replace: dict={}):
        self.args_order = ["key", "value", "exclude"]
        self._logger = logger
        self.prefix_replace = prefix_replace

    def __getattr__(self, name):
        return getattr(self._logger, name)

    def record(self, *args, **kwargs):
        for i, arg in enumerate(args):
            kwargs[self.args_order[i]] = arg

        key_split = kwargs["key"].split("/")
        prefix = key_split[0]
        if prefix in self.prefix_replace.keys():
            key_split[0] = self.prefix_replace[prefix]
            kwargs["key"] = "/".join(key_split)

        self._logger.record(**kwargs)
