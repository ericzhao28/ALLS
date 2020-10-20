from comet_ml.api import API, APIExperiment
from config import comet_ml_key

comet_api = API(api_key=comet_ml_key)
for key in ["405", "301", "1000"]:
    exps = comet_api.get_experiments(
        "ericzhao28",
        project_name="active-label-shift-adaptation",
        pattern=".*" + key + ".*")
    for exp in exps:
        if exp.get_metadata()["archived"]:
            continue
        if key in exp.get_name():
            print(exp.get_name())
            x = comet_api.get("ericzhao28/active-label-shift-adaptation/" + exp.get_metadata()["experimentKey"])
            for asset in x.get_asset_list():
                assetid = asset["assetId"]
                print(asset["fileName"])
                if "csv" not in asset["fileName"]:
                    continue
                with open("../data/" + exp.get_name() + ".csv", "w") as f:
                    f.write(exp.get_asset(asset_id=assetid, return_type="text"))
