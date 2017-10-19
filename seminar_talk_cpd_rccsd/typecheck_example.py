import inspect


def checkargs(function):
    def _f(*arguments):
        for index, argument in enumerate(inspect.getfullargspec(function)[0]):
            if not isinstance(arguments[index], function.__annotations__[argument]):
                raise TypeError("{} is not of type {}".format(
                    arguments[index], function.__annotations__[argument]))
        return function(*arguments)
    _f.__doc__ = function.__doc__
    return _f


def coerceargs(function):
    def _f(*arguments):
        new_arguments = []
        for index, argument in enumerate(inspect.getfullargspec(function)[0]):
            new_arguments.append(function.__annotations__[
                argument](arguments[index]))
        return function(*new_arguments)
    _f.__doc__ = function.__doc__
    return _f

if __name__ == "__main__":
    @checkargs
    def f(x: int, y: int):
        """
        A doc string!
        """
        return x, y

    @coerceargs
    def g(a: int, b: int):
        """
        Another doc string!
        """
        return a + b

    print(f(1, 2))
    try:
        print(f(3, 4.0))
    except TypeError as e:
        print(e)

    print(g(1, 2))
    print(g(3, 5.1))
