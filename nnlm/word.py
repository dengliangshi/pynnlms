#encoding=utf-8

# ---------------------------------------------------------Libraries--------------------------------------------------------
# Standard Libraries

# Third-party Libraries

# User Define Modules

# --------------------------------------------------------Global Strings----------------------------------------------------

# ----------------------------------------------------------Class Main------------------------------------------------------
class Word(object):
    """A word in vocabulary.
    """
    def __init__(self, name, index, cindex):
        """Initialization function.
        :Param name: this word string
        :Param index: index of this word in vocabulary
        :Param cindex: class index of this word
        """
        self.name = name
        self.index = index
        self.cindex = cindex

    def __repr__(self):
        """Instance display format.
        """
        return '<Word: %s>' % self.name