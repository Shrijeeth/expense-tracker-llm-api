from ludwig.api import LudwigModel

def load_model():
    print('***: ', )
    model = LudwigModel.load('./src/models/expense_tracker_llm')
    print(model)
    return model