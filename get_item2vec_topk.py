import json

import torch
from recbole.data import create_dataset, data_preparation
from recbole.utils.case_study import full_sort_topk

from models.Item2Vec import Item2Vec

if __name__ == "__main__":
    model_file = "your model file path(.pth)"

    checkpoint = torch.load(model_file)
    config = checkpoint["config"]

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = Item2Vec(config, train_data.dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    ground_list = []
    uid_list = []
    for batch_idx, batched_data in enumerate(test_data):
        interaction, row_idx, positive_u, positive_i = batched_data
        ground_list.append([int(v) for v in positive_i.numpy().tolist()])
        uid_list.append(interaction.user_id.numpy()[0])

    # topk predict
    topk_score, topk_iid_list = full_sort_topk(
        uid_list, model, test_data, k=10, device="cpu"
    )
    ranked_list = topk_iid_list.cpu()

    all_results = {}
    for uid, g_list, r_list in zip(uid_list, ground_list, ranked_list):
        external_uid = dataset.id2token(dataset.uid_field, uid)
        all_results[external_uid] = {
            "ground_list_id": [v for v in dataset.id2token(dataset.iid_field, g_list)],
            "predict_list_id": [v for v in dataset.id2token(dataset.iid_field, r_list)],
            "ground_list": [v for v in dataset.id2token("product_name", g_list)],
            "predict_list": [v for v in dataset.id2token("product_name", r_list)],
        }

    text = json.dumps(all_results, sort_keys=True, ensure_ascii=False, indent=2)
    with open("item2vec_results.json", "w") as fh:
        fh.write(text)
