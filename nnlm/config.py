#encoding utf-8

# ---------------------------------------------------------Libraries--------------------------------------------------------
# Standard Libraries
import os

# Third-party Libraries


# User Defined Modules


# --------------------------------------------------------Global Strings----------------------------------------------------


# -------------------------------------------------------------Main---------------------------------------------------------
class Config(dict):
    """Configuration
    """
    def __init__(self, defaults=None):
        dict.__init__(self, defaults or {})

    def from_object(self, obj):
        """Updates the values from the given object.
        :param obj: an import name or object.
        """
        for key in dir(obj):
            if key.isupper():
                self[key] = getattr(obj, key)

    def from_dict(self, obj):
        """Updates the values from the given dict.
        :param obj: an dict.
        """
        for key, value in obj.items():
            if key.isupper():
                self[key] = value

    def from_json(self, filename):
        """Updates the values in the config from a JSON file. 
        :Param filename: the filename of the JSON file.  
        """
        filename = os.path.join(self.root_path, filename)
        try:
            with open(filename) as json_file:
                obj = json.loads(json_file.read())
        except IOError as e:
            e.strerror = 'Unable to load configuration file (%s)' % e.strerror
            raise
        return self.from_mapping(obj)

    def from_mapping(self, *mapping, **kwargs):
        """Updates the config like :meth:`update` ignoring items with non-upper keys.
        """
        mappings = []
        if len(mapping) == 1:
            if hasattr(mapping[0], 'items'):
                mappings.append(mapping[0].items())
            else:
                mappings.append(mapping[0])
        elif len(mapping) > 1:
            raise TypeError(
                'expected at most 1 positional argument, got %d' % len(mapping)
            )
        mappings.append(kwargs.items())
        for mapping in mappings:
            for (key, value) in mapping:
                if key.isupper():
                    self[key] = value
        return True

    def __repr__(self):
        """Instance display format.
        """
        return '<%s %s>' % (self.__class__.__name__, 
            dict.__repr__(self))