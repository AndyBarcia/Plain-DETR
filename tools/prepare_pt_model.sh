mkdir ./pt_models

# MIM pre-trained model, swinv2-small
echo "Downloading MIM pre-trained model..."
wget -O ./pt_models/swinv2_small_1k_500k_mim_pt.pth \
"https://huggingface.co/zdaxie/SimMIM/resolve/main/simmim_swinv2_pretrain_models/swinv2_small_1kper10_500k.pth?download=true"

# Supervised pre-trained model, swinv2-small
#echo "Downloading supervised pre-trained model..."
#wget -O ./pt_models/swinv2_small_patch4_window16_256_sup_pt.pth \
#"https://msravcghub.blob.core.windows.net/plaindetr-release/pretrain_models/swinv2_small_patch4_window16_256.pth?sv=2020-04-08&st=2023-11-11T08%3A02%3A36Z&se=2049-12-31T08%3A02%3A00Z&sr=b&sp=r&sig=7SLK7y6TOSXqw5zSWJiCwTEVnDun6u2IYImutw7yDHQ%3D"