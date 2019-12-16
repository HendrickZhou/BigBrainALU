class ModelFactory(object):
    def __init__(self):
        self._model_dict = dict()
    
    def __call__(self, name):
        """
        return an Object of concrete model
        """
        if name.upper() not in self._model_dict:
            raise Exception("model name not registered, plz check if there's a typo")
        print("creating model object:{}".format(name.upper()))
        return self._model_dict[name]()

    def register(self, name, model_init_f):
        """
        receive a model's creating function
        Pros: create the object for registering in one place, avoid the object leaking everywhere
            : lazy initializing. Not created until the factory is called
        save the function to our dictionary
        """
        self._model_dict[name.upper()] = (model_init_f)

    def get_registed_models(self):
        return self._model_dict.values()

_global_factory = ModelFactory()
def get_global_factory():
    return _global_factory


class ModelRegister(object):
    """
    Used as decorator on each model's script
    A creating function in script is required

    use case: in cap_100.py:

    @factory.ModelRegister("cap_100")
    def create_model():
        return Cap100()
    """
    def __init__(self, name):
        self.name = name.upper()

    def __call__(self, model_init_f):
        _global_factory.register(self.name, model_init_f)
