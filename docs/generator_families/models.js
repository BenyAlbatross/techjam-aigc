const m = (name, year, family, era, status, architecture, question, lineage, implication, sources) => ({name, year, family, era, status, architecture, question, lineage, implication, sources});

const models = [
  m("GAN / DCGAN", "2014–2015", "GAN", "foundation", "paper + code lineage",
    "A generator maps noise to pixels while a discriminator distinguishes generated from real samples. DCGAN made the idea reusable for vision with strided convolution, transposed convolution, batch normalization, and a stable architectural recipe.",
    "Can an implicit neural generator learn a realistic distribution without specifying a pixel likelihood—and can convolution make the adversarial game stable enough to train?",
    "The 2014 GAN objective is the root. DCGAN leads into ProGAN, BigGAN, StyleGAN, conditional GANs, and most translation GANs.",
    "Classic detectors often exploit one-pass learned-upsampling traces. Those cues should not be assumed to survive alias-free or modern text-GAN designs.",
    [["GAN paper","https://arxiv.org/abs/1406.2661"],["DCGAN","https://arxiv.org/abs/1511.06434"]]),

  m("pix2pix / CycleGAN", "2016–2017", "GAN", "foundation", "open research",
    "pix2pix is a conditional encoder-decoder GAN trained on paired images with adversarial and reconstruction losses. CycleGAN uses two generators/discriminators plus cycle consistency to learn translation from unpaired domains.",
    "How can a GAN translate an input into another visual domain while preserving content, especially without paired examples?",
    "Conditional GAN → pix2pix paired translation → CycleGAN unpaired translation. StarGAN later expands two domains to many.",
    "These outputs inherit scene structure from real input, so they are edited/translated rather than purely generated. Keep them as an adjacent stress test.",
    [["pix2pix","https://arxiv.org/abs/1611.07004"],["CycleGAN","https://arxiv.org/abs/1703.10593"]]),

  m("ProGAN", "2017", "GAN", "foundation", "open research",
    "A convolutional GAN whose generator and discriminator grow progressively from low resolution to 1024². Equalized learning rates, pixel normalization, and minibatch statistics improve stability and variation.",
    "Can high-resolution GAN training be stabilized by learning coarse structure before fine detail?",
    "DCGAN-style convolution → progressive growing. It directly precedes StyleGAN and is a common fake-detection training source.",
    "Narrow-domain ProGAN data can teach face, alignment, or codec shortcuts. Break generator–dataset coupling.",
    [["Paper","https://arxiv.org/abs/1710.10196"],["Code","https://github.com/tkarras/progressive_growing_of_gans"]]),

  m("StarGAN / StarGAN v2", "2017–2019", "GAN", "foundation", "open research · EDA",
    "One conditional generator translates an input across multiple domains. The first model uses target labels; v2 adds a style encoder and mapping network so each target domain can produce diverse appearances.",
    "Can one image translator replace a separate model for every domain pair while retaining diversity?",
    "pix2pix/CycleGAN translation → unified StarGAN → style-diverse StarGAN v2.",
    "An input is preserved and restyled, so this sits near the challenge boundary rather than squarely inside pure noise-to-image generation.",
    [["StarGAN","https://arxiv.org/abs/1711.09020"],["v2","https://arxiv.org/abs/1912.01865"]]),

  m("BigGAN", "2018", "GAN", "foundation", "open research · EDA",
    "A large class-conditional residual GAN trained on ImageNet with spectral normalization, self-attention, large batches, orthogonal regularization, and a truncation trick that trades diversity for fidelity.",
    "Does scaling GAN capacity and batch size unlock high-fidelity synthesis on a broad multi-class dataset, and what instabilities appear?",
    "ResNet GAN + SAGAN/self-attention + large-scale training; a branch away from face-centric ProGAN/StyleGAN.",
    "A key holdout between narrow-domain style GANs and broad ImageNet GANs. Low native resolution can become a shortcut after resizing.",
    [["Paper","https://arxiv.org/abs/1809.11096"],["Code","https://github.com/deepmind/deepmind-research/tree/master/biggan"]]),

  m("StyleGAN 1 → 3 / StyleGAN-XL", "2018–2022", "GAN", "bridge", "open research · EDA",
    "StyleGAN maps noise into an intermediate style space that modulates synthesis layers. StyleGAN2 redesigns normalization; StyleGAN3 makes synthesis alias-free; StyleGAN-XL scales StyleGAN3 to diverse ImageNet data using pretrained projected discriminators.",
    "Can high-resolution generation expose editable controls, remove texture sticking, and scale beyond aligned faces without losing fidelity?",
    "ProGAN → style-based StyleGAN → cleaner StyleGAN2 → alias-free StyleGAN3 → diverse-data StyleGAN-XL.",
    "Version and training domain are essential. StyleGAN3 deliberately removes some artifacts on which older detectors relied.",
    [["StyleGAN","https://arxiv.org/abs/1812.04948"],["StyleGAN2","https://arxiv.org/abs/1912.04958"],["StyleGAN3","https://arxiv.org/abs/2106.12423"],["StyleGAN-XL","https://arxiv.org/abs/2202.00273"]]),

  m("DF-GAN", "2020", "GAN", "bridge", "open research · EDA",
    "A one-stage text-to-image GAN with deep text-image fusion and a target-aware discriminator trained with matching-aware gradient penalty. It avoids earlier multi-generator stacks.",
    "Can text-to-image GANs become simpler and more aligned without auxiliary matching networks or stacked generators?",
    "StackGAN/AttnGAN conditioning → simpler one-stage adversarial generator. GALIP later injects CLIP semantics.",
    "Adds text conditioning without iterative diffusion; useful for testing mechanics rather than semantic quality.",
    [["Paper","https://arxiv.org/abs/2008.05865"],["Code","https://github.com/tobran/DF-GAN"]]),

  m("GALIP", "2023", "GAN", "bridge", "open research · EDA",
    "A text-to-image GAN using pretrained CLIP inside both generator and discriminator. Bridge features and learnable prompts transfer vision-language knowledge while synthesis remains one-pass.",
    "Can a small fast GAN inherit a foundation model’s scene understanding and compete with much larger autoregressive or diffusion systems?",
    "DF-GAN-style text GAN + CLIP priors: architecturally GAN, semantically foundation-model-assisted.",
    "A warning against simplistic fingerprints: CLIP changes semantics and data needs even though the renderer is adversarial.",
    [["Paper","https://arxiv.org/abs/2301.12959"],["Code","https://github.com/tobran/GALIP"]]),

  m("GigaGAN", "2023", "GAN", "bridge", "research · EDA",
    "A scaled text-conditioned GAN with a StyleGAN-like hierarchy, text conditioning throughout, and a separate efficient upsampler. It creates 512px images in one generator pass and can upsample far higher.",
    "After diffusion became dominant, can GANs scale to web text-image data while retaining speed, editing, and high-resolution advantages?",
    "StyleGAN and text-GAN ideas scaled to LAION-like diversity; a counterexample to ‘all modern T2I is diffusion.’",
    "A critical unseen-GAN test. ProGAN/StyleGAN2 artifacts may not transfer to its modern layers and upsampler.",
    [["Paper","https://arxiv.org/abs/2303.05511"],["Project","https://mingukkang.github.io/GigaGAN/"]]),

  m("VAE", "2013", "VAE / tokens", "foundation", "foundational paper",
    "A probabilistic encoder maps an image to a continuous latent distribution and a decoder reconstructs or samples pixels. The reparameterization trick permits end-to-end variational learning.",
    "Can a neural latent-variable model learn a smooth sampleable representation with tractable approximate inference?",
    "Classical latent variables → neural amortized inference. VQ models discretize this latent; latent diffusion uses an autoencoder chiefly as compressor.",
    "A VAE may be the full generator or only diffusion’s first/last stage, so decoder traces can recur across many checkpoints.",
    [["Paper","https://arxiv.org/abs/1312.6114"]]),

  m("VQ-VAE / VQ-VAE-2", "2017–2019", "VAE / tokens", "foundation", "open research · EDA",
    "The encoder chooses vectors from a learned discrete codebook; the decoder reconstructs pixels; a separate autoregressive prior models code sequences. VQ-VAE-2 adds hierarchical latent maps.",
    "Can discrete codes prevent posterior collapse and separate representation learning from prior modeling?",
    "Continuous VAE → vector quantization → hierarchical VQ-VAE-2. DALL·E 1, VQGAN, MaskGIT, and MAGE inherit image-as-tokens.",
    "The codebook and decoder are a shared formation bottleneck. Interpret prior and decoder artifacts separately.",
    [["VQ-VAE","https://arxiv.org/abs/1711.00937"],["VQ-VAE-2","https://arxiv.org/abs/1906.00446"]]),

  m("VQGAN + transformer", "2020", "VAE / tokens", "bridge", "open research · EDA",
    "A vector-quantized autoencoder uses perceptual and adversarial losses for aggressive, visually faithful compression. A transformer autoregressively models the shortened discrete-code sequence.",
    "Can perceptual/adversarial compression make transformer-based high-resolution image generation tractable?",
    "VQ-VAE codes + GAN/perceptual reconstruction + autoregressive prior; a bridge to token generators and latent diffusion.",
    "A hybrid: the prior is autoregressive but the decoder is adversarially trained. A single family label loses information.",
    [["Paper","https://arxiv.org/abs/2012.09841"],["Code","https://github.com/CompVis/taming-transformers"]]),

  m("DALL·E 1", "2021", "Autoregressive", "bridge", "closed weights · published",
    "A 12B decoder-only transformer receives text followed by a 32×32 grid of discrete image tokens from a discrete VAE and predicts the stream autoregressively. CLIP reranks candidates.",
    "Can text-to-image be framed as large-scale language modeling and generalize zero-shot beyond fixed classes?",
    "VQ-VAE-like tokenizer + GPT-style causal transformer + CLIP reranking. DALL·E 2 changes decoder family rather than merely scaling it.",
    "A canonical discrete autoregressive family; important precisely because DALL·E 2 is not the same mechanism.",
    [["OpenAI","https://openai.com/index/dall-e/"],["Paper","https://arxiv.org/abs/2102.12092"]]),

  m("VQ-Diffusion (VQDM)", "2021", "VAE / tokens", "bridge", "open research · EDA",
    "A VQ-VAE turns images into discrete codes; a conditional transformer corrupts/restores them with mask-and-replace categorical diffusion rather than Gaussian noise on RGB or continuous latents.",
    "Can bidirectional discrete diffusion avoid left-to-right error accumulation and slow autoregressive decoding?",
    "VQ tokens + DDPM-inspired categorical transitions; the improved version adds classifier-free guidance and better sampling.",
    "It shares diffusion terminology but not the continuous denoising path of DDPM, ADM, or Stable Diffusion.",
    [["Paper","https://arxiv.org/abs/2111.14822"],["Improved","https://arxiv.org/abs/2205.16007"]]),

  m("MAE", "2021", "VAE / tokens", "bridge", "representation model · EDA caveat",
    "An asymmetric Vision Transformer autoencoder observes a small subset of image patches and uses a lightweight decoder to reconstruct masked pixels. It was designed for representation learning, not general text-to-image generation.",
    "Can high mask ratios turn reconstruction into an efficient scalable visual pretraining task?",
    "BERT-style masking applied to image patches. MAGE and masked token generators turn the concept into actual generation.",
    "WildFake’s MAE label should mean masked reconstruction/completion, not a peer of SD or BigGAN.",
    [["Paper","https://arxiv.org/abs/2111.06377"],["Code","https://github.com/facebookresearch/mae"]]),

  m("MaskGIT", "2022", "VAE / tokens", "bridge", "open research",
    "A bidirectional transformer repeatedly predicts masked discrete image tokens in parallel, committing the most confident tokens over a small number of refinement iterations.",
    "Can masked parallel prediction generate much faster than raster-order transformers while using full context?",
    "VQ tokenization + BERT-style masking; motivates MAGE and Google’s Muse.",
    "Its discrete confidence-ordered refinement is not Gaussian denoising. Keep it in a token-family holdout.",
    [["Paper","https://arxiv.org/abs/2202.04200"],["Project","https://masked-generative-image-transformer.github.io/"]]),

  m("MAGE", "2022", "VAE / tokens", "bridge", "open research · EDA",
    "A ViT consumes semantic VQGAN tokens. Variable mask ratios let one network reconstruct partially visible images and generate from nearly all-masked token grids.",
    "Can one masked model unify strong self-supervised representations and image synthesis?",
    "MAE masking + MaskGIT generation + VQGAN tokens. MAGE is generative; MAE is primarily representational.",
    "Hold it out from both continuous SD latents and causal DALL·E-style raster generation.",
    [["Paper","https://arxiv.org/abs/2211.09117"],["Code","https://github.com/LTH14/mage"]]),

  m("Parti", "2022", "Autoregressive", "bridge", "research · closed weights",
    "An encoder-decoder transformer maps text to discrete image tokens from a ViT-VQGAN tokenizer. It scales to 20B parameters and reranks candidate samples.",
    "Do language-model scaling laws transfer to content-rich text-to-image generation?",
    "DALL·E-style token modeling with an encoder-decoder transformer and systematic scaling; introduces PartiPrompts.",
    "Semantic competence can improve without diffusion. Formation family and output quality are separate axes.",
    [["Paper","https://arxiv.org/abs/2206.10789"],["Project","https://sites.research.google/parti/"]]),

  m("Muse (Google research)", "2023", "VAE / tokens", "bridge", "research · closed weights",
    "A text-conditioned masked transformer predicts discrete image tokens in parallel over refinement steps, using a frozen T5 language encoder and super-resolution stage.",
    "Can masked token modeling retain strong alignment while being faster than causal transformers and diffusion?",
    "MaskGIT + T5 conditioning + VQ decoder. Unrelated to Meta’s later product using the Muse name.",
    "Preserve vendor in metadata: ‘Muse’ has a real name collision.",
    [["Paper","https://arxiv.org/abs/2301.00704"]]),

  m("DDPM", "2020", "Pixel diffusion", "foundation", "open research · EDA",
    "A U-Net predicts noise or score at random timesteps. Generation starts from Gaussian noise and applies many learned reverse transitions until an RGB image emerges.",
    "Can diffusion models deliver high-quality images with a stable likelihood-related objective competitive with GANs?",
    "Diffusion probabilistic modeling + denoising score matching → practical DDPM. ADM, Imagen, LDM, and DiT inherit the framework.",
    "DDPM is an objective/process, not one dataset or network. Preserve architecture, schedule, sampler, steps, and resolution.",
    [["Paper","https://arxiv.org/abs/2006.11239"],["Code","https://github.com/hojonathanho/diffusion"]]),

  m("DDIM", "2020", "Pixel diffusion", "foundation", "sampler/formulation · EDA",
    "A non-Markovian formulation with the same training objective as DDPM. The reverse process can be deterministic and use fewer steps; ‘DDIM’ often names a sampler for another trained denoiser.",
    "Can the expensive DDPM sampling chain be accelerated and made inversion-friendly without retraining?",
    "DDPM training → alternative implicit trajectories → standard sampler/inversion tool in latent diffusion.",
    "Do not count DDIM as a clean architecture family. Without the underlying checkpoint, the EDA label is incomplete.",
    [["Paper","https://arxiv.org/abs/2010.02502"],["Code","https://github.com/ermongroup/ddim"]]),

  m("ADM", "2021", "Pixel diffusion", "foundation", "open research · EDA",
    "An improved pixel-space diffusion U-Net with residual blocks, multi-resolution attention, adaptive normalization, learned variance, and optional classifier guidance.",
    "Can better architecture and guidance let diffusion surpass class-conditional GANs while retaining mode coverage?",
    "Improved DDPM → architectural scaling + classifier guidance; predecessor to text-guided pixel diffusion.",
    "Its class conditioning and pixel-space decoder make it different from Stable Diffusion despite shared denoising.",
    [["Paper","https://arxiv.org/abs/2105.05233"],["Code","https://github.com/openai/guided-diffusion"]]),

  m("GLIDE", "2021", "Pixel diffusion", "bridge", "open small model",
    "A text-conditioned pixel diffusion U-Net followed by diffusion upsampling. It compares CLIP guidance with classifier-free guidance and supports inpainting.",
    "How should language guide diffusion for photorealism, caption fidelity, and editing?",
    "ADM-style pixel diffusion + text + cascade. DALL·E 2’s decoder follows the same broad OpenAI branch.",
    "Record whether samples come from the released filtered model or the unreleased large system.",
    [["Paper","https://arxiv.org/abs/2112.10741"],["Code","https://github.com/openai/glide-text2im"]]),

  m("DALL·E 2 / unCLIP", "2022", "Pixel diffusion", "bridge", "published · EDA/API legacy",
    "A prior predicts a CLIP image embedding from text; a diffusion decoder inverts it into an image; diffusion super-resolution follows. The research result favored a diffusion prior.",
    "Can CLIP’s semantic image space improve diversity, variations, and text-guided manipulation?",
    "Replaces DALL·E 1’s causal decoder with CLIP-latent prior + GLIDE-like diffusion stack.",
    "DALL·E 1, 2, 3, and GPT Image span different disclosed mechanisms. Never merge by brand.",
    [["Paper","https://arxiv.org/abs/2204.06125"],["OpenAI","https://openai.com/index/hierarchical-text-conditional-image-generation-with-clip-latents/"]]),

  m("Imagen 1 → 4", "2022–2025", "Pixel diffusion", "bridge", "API lineage · deprecated",
    "Published Imagen 1 uses frozen T5-XXL, a 64×64 pixel diffusion model, and two super-resolution diffusion models. Google calls later Imagen releases diffusion models but does not disclose every version’s full internals.",
    "Does language understanding drive alignment more than a larger decoder, and can cascades deliver photorealism?",
    "Cascaded diffusion + large frozen language encoder. Imagen 3/4 improve quality, text, and speed; Gemini native image models replaced them in the API.",
    "Keep versions/API dates separate. Google adds SynthID, but the challenge forbids watermark reliance.",
    [["Research","https://imagen.research.google/"],["Imagen 4","https://deepmind.google/models/imagen/"],["Deprecation","https://ai.google.dev/gemini-api/docs/deprecations"]]),

  m("Latent Diffusion / Stable Diffusion 1.x", "2021–2022", "Latent diffusion", "bridge", "open weights · EDA",
    "A perceptual autoencoder compresses RGB; a convolutional U-Net denoises the continuous latent; cross-attention injects CLIP text; the frozen decoder maps back to pixels. SD 1.x packages the recipe at 512px.",
    "Can diffusion move into a compact perceptual latent while preserving detail and flexible conditioning?",
    "VQGAN/autoencoder compression + DDPM U-Net + cross-attention. Stable Diffusion operationalizes the LDM paper at LAION scale.",
    "The shared VAE decoder is a bottleneck across many checkpoints. Hold out VAEs and non-SD families.",
    [["LDM paper","https://arxiv.org/abs/2112.10752"],["SD release","https://stability.ai/news-updates/stable-diffusion-public-release"],["Code","https://github.com/CompVis/stable-diffusion"]]),

  m("Stable Diffusion 2.x / Wukong", "2022", "Latent diffusion", "bridge", "open weights",
    "SD 2.x keeps the latent U-Net pipeline but changes text encoder, data/filtering, resolutions, and configurations. Wukong retargets the broad open LDM ecosystem to Chinese text-image data.",
    "Can open latent diffusion improve resolution, change data/safety choices, and cover other languages?",
    "Direct LDM/SD 1.x descendants rather than new top-level architectures, but distinct data/checkpoint domains.",
    "Near-relative holdouts reveal checkpoint overfit. Wukong should not be silently merged with SD 1.4.",
    [["SD 2 code","https://github.com/Stability-AI/stablediffusion"],["Wukong","https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/Wukong-Huahua"]]),

  m("SDXL", "2023", "Latent diffusion", "bridge", "open weights · EDA",
    "A latent U-Net roughly three times larger than earlier SD, with more attention, two text encoders, size/crop micro-conditioning, multi-aspect training, and an optional refiner.",
    "How far can the established latent U-Net recipe scale before moving to transformers?",
    "SD 1/2 → larger base U-Net + richer conditioning + optional refiner; last major U-Net SD before SD3’s MMDiT break.",
    "Base-only, refiner, Turbo, Lightning, LoRA, and ControlNet are documented subdomains, not identical outputs.",
    [["Paper","https://arxiv.org/abs/2307.01952"],["Code","https://github.com/Stability-AI/generative-models"]]),

];

const catalog = document.querySelector("#catalog");
const search = document.querySelector("#search");
const familyFilter = document.querySelector("#family-filter");
const eraFilter = document.querySelector("#era-filter");
const count = document.querySelector("#count");
const noResults = document.querySelector("#no-results");
const escapeHtml = (value) => String(value).replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));

[...new Set(models.map(x => x.family))].sort().forEach(family => {
  const option = document.createElement("option");
  option.value = family;
  option.textContent = family;
  familyFilter.append(option);
});

function render() {
  const query = search.value.trim().toLowerCase();
  const filtered = models.filter(model => {
    const text = Object.values(model).flat(2).join(" ").toLowerCase();
    return (!query || text.includes(query)) && (familyFilter.value === "all" || model.family === familyFilter.value) && (eraFilter.value === "all" || model.era === eraFilter.value);
  });
  catalog.innerHTML = filtered.map(model => `<article class="model-card" data-family="${escapeHtml(model.family)}"><div class="model-head"><div class="model-title"><h3>${escapeHtml(model.name)}</h3><span class="year">${escapeHtml(model.year)}</span></div><div class="badges"><span class="badge">${escapeHtml(model.family)}</span><span class="badge status">${escapeHtml(model.status)}</span></div></div><dl><dt>Architecture</dt><dd>${escapeHtml(model.architecture)}</dd><dt>Research question</dt><dd>${escapeHtml(model.question)}</dd><dt>Lineage</dt><dd>${escapeHtml(model.lineage)}</dd></dl><p class="implication"><strong>Detector lens.</strong> ${escapeHtml(model.implication)}</p><p class="sources">${model.sources.map(([label,url]) => `<a href="${escapeHtml(url)}">${escapeHtml(label)}</a>`).join("")}</p></article>`).join("");
  count.textContent = `${filtered.length} ${filtered.length === 1 ? "entry" : "entries"}`;
  noResults.style.display = filtered.length ? "none" : "block";
}

search.addEventListener("input", render);
familyFilter.addEventListener("change", render);
eraFilter.addEventListener("change", render);
render();
