"""Apply deterministic post-envelope outlier detection to the latest 178 result."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .point_validity import detect_valid_points


def _plot(item, final, outlier, output, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f=np.asarray(item["frequency"],float); z=np.asarray(item["z_real"])+1j*np.asarray(item["z_imag"]); inside=np.asarray(item["ml_frequency_active_mask"],bool); x=np.log10(f)
    fig,axes=plt.subplots(1,3,figsize=(15,4)); axes[0].scatter(x,z.real,s=8,color="0.75",label="raw"); axes[0].scatter(x[inside],z.real[inside],s=12,color="tab:blue",label="ML range"); axes[0].scatter(x[outlier],z.real[outlier],marker="x",color="tab:red",s=28,label="outlier"); axes[0].scatter(x[final],z.real[final],s=14,color="tab:green",label="final"); axes[0].set_title("Re(Z)")
    axes[1].scatter(x,z.imag,s=8,color="0.75"); axes[1].scatter(x[final],z.imag[final],s=14,color="tab:green"); axes[1].set_title("Im(Z)")
    axes[2].scatter(z.real,-z.imag,s=8,color="0.75"); axes[2].scatter(z.real[final],-z.imag[final],s=14,color="tab:green"); axes[2].set_title("Nyquist")
    for ax in axes[:2]: ax.axvline(np.log10(item["predicted_f_min"]),color="tab:red",linestyle="--"); ax.axvline(np.log10(item["predicted_f_max"]),color="tab:purple",linestyle="--"); ax.set_xlabel("log10(f)")
    axes[0].legend(fontsize=8); fig.suptitle(label); fig.tight_layout(); fig.savefig(output/f"{label}.png",dpi=130); plt.close(fig)


def continue_inference(source: Path, output: Path, threshold: float = 4.0):
    started=time.perf_counter(); payload=json.loads(source.read_text(encoding="utf-8")); root=payload.get("ml_results",payload); spectra=root.get("spectra",[]); output.mkdir(parents=True,exist_ok=True); plots=output/"plots"; plots.mkdir(exist_ok=True); results=[]; rows=[]
    for index,item in enumerate(spectra):
        f=np.asarray(item["frequency"],float); z=np.asarray(item["z_real"])+1j*np.asarray(item["z_imag"]); inside=np.asarray(item["ml_frequency_active_mask"],bool); stage1=np.asarray(item.get("stage1_active_mask",np.ones(f.size,bool)),bool)
        detected,score,diagnostics=detect_valid_points(f,z,threshold=threshold,neighborhood=3,min_points=4,frequency_range=(float(item["predicted_f_min"]),float(item["predicted_f_max"])),max_iterations=2,return_diagnostics=True)
        outlier=inside & (np.asarray(diagnostics["rejection_reason"],dtype=object)=="local_anomaly"); final=inside & stage1 & ~outlier
        result=dict(item); result.update({"stage2_active_mask":detected.tolist(),"deterministic_outlier_mask":outlier.tolist(),"deterministic_outlier_score":score.tolist(),"final_ml_active_mask":final.tolist(),"stage2_threshold":threshold,"stage2_applied":True,"predicted_process_count":None,"process_prediction_confidence":None,"predicted_L0_required":None,"L0_prediction_confidence":None,"L0_prediction_status":"unavailable_no_serialized_model","suggested_EEC":None,"topology_prediction_status":"unavailable_no_serialized_model"}); results.append(result)
        rows.append({"spectrum_id":item.get("spectrum_id"),"source_name":item.get("source_name"),"cycle":item.get("cycle"),"voltage":item.get("voltage"),"current":item.get("current"),"Time":item.get("metadata",{}).get("Time",item.get("time")),"Cycle mod 15":item.get("metadata",{}).get("Cycle mod 15"),"predicted_f_min":item.get("predicted_f_min"),"predicted_f_max":item.get("predicted_f_max"),"n_raw_points":len(f),"n_frequency_selected":int(inside.sum()),"n_outliers":int(outlier.sum()),"n_final_active":int(final.sum()),"predicted_process_count":None,"P_ONE_PROCESS":None,"P_TWO_PROCESS":None,"predicted_L0_required":None,"P_L0_REQUIRED":None,"P_L0_NOT_REQUIRED":None,"suggested_EEC":None})
        if index<8: _plot(item,final,outlier,plots,f"spectrum_{index+1:03d}")
    out_payload={"ml_results":{**{k:v for k,v in root.items() if k!="spectra"},"schema_version":"4.0","source_file":str(root.get("source_file",source)),"stage2_configuration":{"detector":"ml.point_validity.detect_valid_points","threshold":threshold,"frequency_range_applied_before_detection":True},"topology_prediction_status":"unavailable_no_serialized_topology_model","L0_prediction_status":"unavailable_no_serialized_L0_model","spectra":results}}; json_path=output/"178_ML_preprocessed_results.eisfit.json"; json_path.write_text(json.dumps(out_payload,indent=2),encoding="utf-8"); pd.DataFrame(rows).to_csv(output/"178_ML_EEC_predictions.csv",index=False); report={"source_result":str(source),"output_json":str(json_path),"spectra":len(results),"total_points":int(sum(len(x["frequency"]) for x in results)),"frequency_removed":int(sum(len(x["frequency"])-sum(x["ml_frequency_active_mask"]) for x in results)),"outliers_removed":int(sum(x["n_outliers"] for x in rows)),"final_active_points":int(sum(x["n_final_active"] for x in rows)),"stage2_threshold":threshold,"topology_model":"unavailable_no_serialized_model","L0_model":"unavailable_no_serialized_model","retrained":False,"bayes_drt2":False,"runtime_s":time.perf_counter()-started}; (output/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2)); return report


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--source",type=Path,default=Path("ml/analysis/unseen_178_new_frequency/178_ML_frequency_results.eisfit.json")); parser.add_argument("--output",type=Path,default=Path("ml/analysis/unseen_178_preprocessed")); parser.add_argument("--threshold",type=float,default=4.0); args=parser.parse_args(); continue_inference(args.source,args.output,args.threshold)


if __name__=="__main__": main()
