from abc import ABCMeta
import inspect

def _get_attributes_names(cls):
    """
    Get attributes names
    """
    attributes = inspect.getmembers(cls, lambda a:not inspect.isroutine(a))
    filtered_attrib = [a for a in attributes if not(a[0].startswith('_') and a[0].endswith('__'))]
    return sorted([p[0] for p in filtered_attrib])


class Historic(metaclass=ABCMeta):
    """
    Class to save variables values along the estimation iterations.

    Parameters
    ----------
    metaclass : by default ABCMeta
    """

    def __init__(self):
        pass


    def get_params(self):
        """
        Create a dictionnary with attributes saved in Historic object. 
        Keys are names and items are corresponding saved variables from the mixture class.

        Returns
        -------
        out
            dict
        """
        out = {}
        for var in _get_attributes_names(self):
            value = getattr(self, var)
            if  hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((var + '__' + k, val) for k, val in deep_items)
            out[var] = value
        return out

    def save_variables(self,values,variable_names):
        """
        Save given attributes in the Historic object, under the name given in variable_names.
        Create a list for each newly saved attribute. 
        If an attribute has already been saved at least once, new value is added to the list.

        Parameters
        ----------
        values : list
            Contains variables to save, can be flots, arrays etc
        variable_names : list
            Contains names to use to save corresponding variables in values list.

        Raises
        ------
        ValueError
            _description_
        """
        if not isinstance(values, list):
            name = variable_names
            if hasattr(self,name):
                getattr(self, name).append(values)
            else :
                setattr(self, name, [])
                getattr(self, name).append(values)
        else :
            if len(values) != len(variable_names):
                raise ValueError("Save variables: Lists should have the same dimension !")

            for name,value in zip(variable_names,values):
                true_name = name
                if hasattr(self,true_name) :
                    getattr(self, true_name).append(value)
                else :
                    setattr(self, true_name, [])
                    getattr(self, true_name).append(value)
