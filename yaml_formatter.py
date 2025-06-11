import logging

import yaml


class YAMLFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.yaml_dumper = yaml.SafeDumper
        self.yaml_dumper.add_representer(str, self._represent_multiline_str)

    def _represent_multiline_str(self, dumper, data):
        if '\n' in data or len(data) > 80:
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
        return dumper.represent_scalar('tag:yaml.org,2002:str', data)

    def format(self, record):
        yaml_output = yaml.dump(
            record,
            Dumper=self.yaml_dumper,
            default_flow_style=False,
            allow_unicode=True,
            width=float('inf'),  # Prevent line wrapping
            indent=2
        )

        return yaml_output.rstrip()
