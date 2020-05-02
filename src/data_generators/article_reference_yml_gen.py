import yaml


class ArticleReferenceGenerator:

    def __init__(self, source_file: str):
        with open(source_file) as file:
            self.__data_list = yaml.safe_load(file)
        self.__pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.__pos < len(self.__data_list):
            result = self.__data_list[self.__pos]
            self.__pos += 1
        else:
            raise StopIteration
        return result

    @property
    def shape(self):
        return len(self.__data_list),
