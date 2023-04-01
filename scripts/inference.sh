# Random Generation
python inference_Full.py --trained_ckpt checkpoints/final_model.pt --results_dir results --identities 3 --size 512 --truncation_ratio 0.7 --no_surface_renderings


# Random Generation, rendering shape
python inference_Full.py --trained_ckpt checkpoints/final_model.pt --results_dir results --identities 3 --size 512 --truncation_ratio 0.7 

# Generating local semantics
# face semantics: 'background','face','eye','brow','mouth','nose','ear','hair','neck+cloth'
python inference_Full.py --trained_ckpt checkpoints/final_model.pt --results_dir results --identities 3 --size 512 --truncation_ratio 0.7 --no_surface_renderings --semantics 2,3