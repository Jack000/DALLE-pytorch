import argparse
from pathlib import Path
from tqdm import tqdm

# torch

import torch

from einops import repeat

# vision imports

from PIL import Image
from torchvision.utils import make_grid, save_image
from torchvision.transforms import functional as TF

# dalle related classes and utils

from dalle_pytorch import DiscreteVAE, OpenAIDiscreteVAE, VQGanVAE, DALLE
from dalle_pytorch.tokenizer import tokenizer, HugTokenizer, YttmTokenizer, ChineseTokenizer

# argument parsing

parser = argparse.ArgumentParser()

parser.add_argument('--dalle_path', type = str, required = True,
                    help='path to your trained DALL-E')

parser.add_argument('--vqgan_model_path', type=str, default = None,
                   help='path to your trained VQGAN weights. This should be a .ckpt file. (only valid when taming option is enabled)')

parser.add_argument('--vqgan_config_path', type=str, default = None,
                   help='path to your trained VQGAN config. This should be a .yaml file.  (only valid when taming option is enabled)')

parser.add_argument('--text', type = str, required = True,
                    help='your text prompt')

parser.add_argument('--num_images', type = int, default = 12, required = False,
                    help='number of images')

parser.add_argument('--batch_size', type = int, default = 4, required = False,
                    help='batch size')

parser.add_argument('--top_k', type = float, default = None, required = False,
                    help='top k filter threshold')

parser.add_argument('--top_p', type = float, default = None, required = False,
                    help='top p filter threshold')

parser.add_argument('--temperature', type = float, default = 1.0, required = False,
                    help='sampling temperature')

parser.add_argument('--outputs_dir', type = str, default = './outputs', required = False,
                    help='output directory')

parser.add_argument('--bpe_path', type = str,
                    help='path to your huggingface BPE json file')

parser.add_argument('--clip_sort', dest='clip_sort', action = 'store_true')

parser.add_argument('--hug', dest='hug', action = 'store_true')

parser.add_argument('--chinese', dest='chinese', action = 'store_true')

parser.add_argument('--taming', dest='taming', action='store_true')

parser.add_argument('--gentxt', dest='gentxt', action='store_true')

args = parser.parse_args()

if args.clip_sort:
    # load OpenAI clip
    import clip
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    clip_model, clip_preprocess = clip.load('ViT-B/16', jit=False)
    clip_model.eval().requires_grad_(False).to(device)

# helper fns

def exists(val):
    return val is not None

# tokenizer

if exists(args.bpe_path):
    klass = HugTokenizer if args.hug else YttmTokenizer
    tokenizer = klass(args.bpe_path)
elif args.chinese:
    tokenizer = ChineseTokenizer()

# load DALL-E

dalle_path = Path(args.dalle_path)

assert dalle_path.exists(), 'trained DALL-E must exist'

load_obj = torch.load(str(dalle_path))
dalle_params, vae_params, weights = load_obj.pop('hparams'), load_obj.pop('vae_params'), load_obj.pop('weights')

dalle_params.pop('vae', None) # cleanup later

if args.taming:
    vae = VQGanVAE(args.vqgan_model_path, args.vqgan_config_path)
elif vae_params is not None:
    vae = DiscreteVAE(**vae_params)
else:
    vae = OpenAIDiscreteVAE()

dalle = DALLE(vae = vae, **dalle_params).cuda()

dalle.load_state_dict(weights)

# generate images

image_size = vae.image_size

texts = args.text.split('|')

for j, text in tqdm(enumerate(texts)):

    text = text.lower()

    if args.gentxt:
        text_tokens, gen_texts = dalle.generate_texts(tokenizer, text=text, filter_thres = args.top_k)
        text = gen_texts[0]
    else:
        text_tokens = tokenizer.tokenize([text], dalle.text_seq_len).cuda()

    text_tokens = repeat(text_tokens, '() n -> b n', b = args.num_images)

    outputs = []

    for text_chunk in tqdm(text_tokens.split(args.batch_size), desc = f'generating images for - {text}'):
        if args.top_k is not None:
            output = dalle.generate_images(text_chunk, temperature=args.temperature, top_k_thresh = args.top_k)
        elif args.top_p is not None:
            output = dalle.generate_images(text_chunk, temperature=args.temperature, top_p_thresh = args.top_p)
        else:
            output = dalle.generate_images(text_chunk, temperature=1.0, top_p_thresh = 0.9)

        outputs.append(output)

    outputs = torch.cat(outputs)

    # save all images
    file_name = text 
    outputs_dir = Path(args.outputs_dir) / file_name.replace(' ', '_')[:(100)]
    outputs_dir.mkdir(parents = True, exist_ok = True)

    if not args.clip_sort:
        for i, image in tqdm(enumerate(outputs), desc = 'saving images'):
            save_image(image, outputs_dir / f'{i}.jpg', normalize=True)
            with open(outputs_dir / 'caption.txt', 'w') as f:
                f.write(file_name)
    else:
        images_sorted = []
        for i, image in enumerate(outputs):
            image = image.add(1).div(2).clamp(0, 1)
            pimg = TF.to_pil_image(image)

            text_features = clip_model.encode_text(clip.tokenize(text).to(device))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            image_features = clip_model.encode_image(clip_preprocess(pimg).unsqueeze(0).to(device))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            similarity = torch.nn.functional.cosine_similarity(image_features, text_features, dim=-1)

            images_sorted.append((image, similarity.item()))

        images_sorted.sort(key=lambda x:x[1], reverse=True)

        for i, image in enumerate(images_sorted):
            save_image(image[0], outputs_dir / f'{i}-{image[1]}.png', normalize=True)

    print(f'created {args.num_images} images at "{str(outputs_dir)}"')
