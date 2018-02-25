from inspect import currentframe, getframeinfo

class tool_debug():
    def get_linenumber():
        cf = currentframe()
        return cf.f_back.f_lineno


    def get_filename():
        cf = currentframe()
        return getframeinfo(cf).filename