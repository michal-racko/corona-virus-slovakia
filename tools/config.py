import yaml

from tools.general import singleton


@singleton
class Config:
    def __init__(self):
        """
        Singleton representing a yaml config file.
        """
        self._config = None

    def read(self,
             file_path: str):
        """
        :param file_path: path to the config file
        """
        try:
            with open(file_path, encoding='utf-8') as config_file:
                self._config = yaml.load(config_file, Loader=yaml.FullLoader)

        except FileNotFoundError:
            raise FileNotFoundError(
                f'Failed to find a config file at: {file_path}'
            )

    def get(self, *args, default=None):
        """
        Can be used to obtain values under multiple keys e.g.

        {
            key0: {
                key1: {
                    key2: value,
                    ...
                    },
                ...
                },
            ...
        }

        by calling get(key0, key1, key2)
        """
        if self._config is None:
            raise Exception('Config file must be read first')

        conf = self._config

        for arg in args:
            if arg not in conf:
                if not default:
                    raise KeyError(
                        f'{"/".join(args)} not found in the config file'
                    )

                else:
                    return default

            conf = conf[arg]

        return conf

    def to_dict(self) -> dict:
        return self._config

    def to_yaml(self) -> str:
        """
        Dumps current configurations into a yaml-serialized string.

        :return:        yaml-serialized config settings
        """
        return yaml.dump(self._config)
