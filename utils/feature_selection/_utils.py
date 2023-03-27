from enum import Enum


def enum_with_attributes_factory(*attributes):
    class EnumWithAttributes(Enum):
        def __new__(cls, value, *args):
            obj = object.__new__(cls)
            obj._value_ = value
            assert len(attributes) == len(args), "Number of attributes must match number of arguments."
            for attr, arg in zip(attributes, args):
                setattr(obj, attr, arg)
            return obj

    return EnumWithAttributes