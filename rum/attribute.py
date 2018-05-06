
from . import core

class Selector:
    '''A class mimicking a function returning whether the feature is OK subject
    to certain criteria. Expects a dict as input.'''

    def __call__(self, attributes):
        return True

    def __repr__(self):
        return self.__class__.__name__
        
class PresenceSelector(Selector):
    '''Selects a feature if it has a specified attribute.'''

    def __init__(self, attribute):
        self.attribute = attribute

    def __call__(self, attributes):
        return attributes.get(self.attribute, None) is not None
        
class ListSelector(Selector):
    '''Selects a feature if it has a specified attribute and its value is
    among the specified values.'''
    
    def __init__(self, attribute, allowed):
        self.attribute = attribute
        self.allowedValues = allowed

    def __call__(self, attributes):
        return attributes.get(self.attribute, None) in self.allowedValues
        
class CompoundSelector(Selector):
    '''Selects a feature according to multiple conditions concatenated with a specified function.'''

    def __init__(self, clauses):
        self.clauses = clauses

class AndSelector(CompoundSelector):
    def __call__(self, attributes):
        for clause in self.clauses:
            if not clause(attributes):
                return False
        return True
    
class OrSelector(CompoundSelector):
    def __call__(self, attributes):
        for clause in self.clauses:
            if clause(attributes):
                return True
        return False
    
class NotSelector(CompoundSelector):
    def __call__(self, attributes):
        for clause in self.clauses:
            return not clause(attributes)
  
COMPOUND_SELECTORS = {'and' : AndSelector, 'or' : OrSelector, 'not' : NotSelector}
  
class Getter:
    '''A class mimicking a function retrieving a value of a certain attribute from a feature attribute dictionary.'''

    def __call__(self, attributes):
        return None
    
    def __repr__(self):
        return self.__class__.__name__
        
    
class AggregateGetter(Getter):
    '''Extracts a single value based on multiple values retrieved by its subgetters.'''

    def __init__(self, getters):
        self.getters = getters

    def __call__(self, attributes):
        values = [getter(attributes) for getter in self.getters]
        try:
            return self.AGGREGATOR(value for value in values if value is not None)
        except ValueError: # empty sequence
            return None
        
class MinGetter(AggregateGetter):
    AGGREGATOR = min

class MaxGetter(AggregateGetter):
    AGGREGATOR = max
    
    
class GatherGetter(Getter):
    '''Tries to extract a value from a number of attributes. Returns the first value found. If nothing is present, returns None.'''

    def __init__(self, names):
        self.names = names
    
    def __call__(self, attributes):
        for name in self.names:
            attrval = attributes.get(name, None)
            if attrval is not None:
                return attrval
        return None

        
class ConstantGetter(Getter):
    '''Always returns a constant value.'''

    def __init__(self, val):
        self.val = val
    
    def __call__(self, attributes):
        return self.val

        
class SwitchGetter(Getter):
    def __init__(self, name, cases):
        self.name = name
        self.cases = cases

    def __call__(self, attributes):
        value = None
        if self.name in attributes:
            value = self.cases.get(attributes[self.name], None)
        return value
      
      
class PresenceGetter(Getter):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __call__(self, attributes):
        return None if (attributes.get(self.name, None) is None) else self.value
            
            
class AttributeGetter(Getter):
    def __init__(self, name):
        self.name = name

    def __call__(self, attributes):
        return attributes.get(self.name, None)
            

class AlternativeGetter(Getter):
    def __init__(self, getters):
        self.getters = getters
    
    def __call__(self, attributes):
        for getter in self.getters:
            value = getter(attributes)
            if value is not None:
                return value
        return None

class MultiGetter(Getter):
    def __init__(self, getters):
        self.getters = getters
    
    def __call__(self, attributes):
        values = []
        for getter in self.getters:
            value = getter(attributes)
            if value is not None:
                values.append(value)
        if len(values) > len(frozenset(values)):
            values = list(frozenset(values))
        return values if values else None

        
class Attribute:
    '''An attribute. Stores its name, type and getter.'''

    TYPES = {'enumstring' : str, 'string' : str, 'float' : float, 'boolean' : bool, 'int' : int}
    UNIT_REMOVER = (lambda x: x.split()[0])
    TYPE_CONV = {str : None, float : UNIT_REMOVER, int : UNIT_REMOVER, bool : None}
    
    def __init__(self, name, type, getter, restrict=False):
        self.name = name # public
        self.type = type # public
        self.typeConverter = self.TYPE_CONV[self.type]
        self.getter = getter
        self.restrict = restrict # public
        
    def get(self, values):
        try:
            val = self.getter(values)
            if val is None:
                return None
            elif type(val) is list:
                return [self.type(item) for item in val]
            elif self.typeConverter is not None and isinstance(val, str):
                return self.type(self.typeConverter(val))
            else:
                return self.type(val)
        except ValueError as mess:
            return None
    
    @classmethod
    def fromConfig(cls, config):
        return cls(config['name'], cls.TYPES[config['type']], cls.parseGetter(config['getter']), restrict=(config.get('selection', None) == 'restrict'))
    
    @classmethod
    def parseGetter(cls, config):
        if 'switch' in config:
            return SwitchGetter(config['switch'], config['cases'])
        elif 'constant' in config:
            return ConstantGetter(config['constant'])
        elif 'ifpresent' in config:
            return PresenceGetter(config['ifpresent'], config['value']) # lambda feat: config['value'] if config['ifpresent'] in feat else None
        elif 'get' in config:
            return AttributeGetter(config['get'])
        elif 'max' in config:
            return MaxGetter([cls.parseGetter(subconfig) for subconfig in config['max']])
        elif 'min' in config:
            return MinGetter([cls.parseGetter(subconfig) for subconfig in config['min']])
        elif 'more' in config:
            return MultiGetter([cls.parseGetter(attrib) for attrib in config['more']])
        elif 'alternative' in config:
            return AlternativeGetter([cls.parseGetter(attrib) for attrib in config['alternative']])
        elif 'gather' in config:
            return GatherGetter(config['gather'])
        else:
            raise core.ConfigError('no getter found: ' + str(config))
        
    def __repr__(self):
        return self.name + '(' + str(self.type) + ',' + str(self.getter) + ')'
    
def selector(config):
    if not config:
        return None
    elif 'compound' in config:
        return COMPOUND_SELECTORS[config['compound']](
            [selector(clause) for clause in config['clauses']]
        )
    else:
        attributeName = next(iter(config.keys()))
        if isinstance(config[attributeName], list):
            return ListSelector(attributeName, config[attributeName])
        elif config[attributeName] == True:
            return PresenceSelector(attributeName)
        else:
            raise core.ConfigError('cannot determine selector type from ' + str(config))

def attribute(config):
    return Attribute.fromConfig(config)