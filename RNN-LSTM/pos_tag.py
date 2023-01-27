from torchtext import data
from torchtext import datasets

#Loading the UD En POS benchmark

TEXT = data.Field(lower = True)
UD_TAGS = data.Field(unk_token = None)
fields = (("text", TEXT), ("udtags", UD_TAGS))
train_data , valid_data , test_data = datasets.UDPOS.splits(fields)

def visualizeSentenceWithTags(example):
    print("Token"+"".join([" "]*(15))+"POS Tag")
    print("---------------------------------")
    for w, t in zip(example['text'], example['udtags']):
        print(w+"".join([" "]*(20- len(w)))+t)

visualizeSentenceWithTags(vars(train_data.examples [997]))