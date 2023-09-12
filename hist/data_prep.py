import numpy as np
import json


def merge_label(label_p, pred_p):
    with open(label_p, 'r') as f:
        labels = json.load(f)
    with open(pred_p, 'r') as f:
        preds = json.load(f)
    for label in labels:
        del label["Patient"]
        if "prediction" in label:
            del label["prediction"]
    labels.extend(
        ({'name': item["name"], "label":item["prediction"] } for item in preds))
    
    return labels


def prep_y_true(slide_names, labels):
    label_map = {item["name"]: item for item in labels}
    y_true = [label_map[slide_name] for idx, slide_name in enumerate(slide_names, 0) 
              if label_map[slide_name].update({"index": idx}) or True] # use side effect to update index
    return y_true

if __name__ == "__main__":
    label_p = "Data/tr_labelled_cases.json"
    pred_p = "Data/tr_pred_cases.json"
    labels = merge_label(label_p, pred_p)
    
    with open("Data/slide_names.json", 'r') as f:
        slide_names = json.load(f)
    
    y_true = prep_y_true(slide_names, labels)

    with open("Data/y_true.json", 'w') as f:
        json.dump(y_true, f)
    