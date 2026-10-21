import numpy as np, pickle, json, os
d="offline_data_hotpot_q1000"
lut=np.load(d+"/pq_codebooks.npy")
pl=pickle.load(open(d+"/posting_lists.pkl","rb"))
order=pickle.load(open(d+"/chunk_ids_order.pkl","rb")); vid={c:i for i,c in enumerate(order)}
R=np.load(d+"/opq_rotation.npy"); E=np.load(d+"/raw_embeddings.npy")
X=(E@R)
Xpq=np.zeros_like(X)
for cid,entries in pl.items():
    for e in entries:
        Xpq[vid[e["chunk_id"]]]=np.concatenate([lut[s,int(e["pq_code"][s])] for s in range(lut.shape[0])])
np.save("X_pq.npy",Xpq.astype(np.float64))
print("X_pq.npy", Xpq.shape, "recon err", np.linalg.norm(X-Xpq,axis=1).mean())
s=json.load(open("h_aimdc_sim/fefet_ref_fit_smu.scaler.json")); s["v_min"]=-0.6553; s["v_max"]=0.6553
json.dump(s,open("h_aimdc_sim/fefet_ref_fit_smu.fs1p31.scaler.json","w"),indent=2)
print("scaler fs1p31 written")
