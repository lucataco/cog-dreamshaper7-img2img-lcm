# Lykon/dreamshaper-7 img2img LCM Cog model

This is an implementation of the [Lykon/dreamshaper-7](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5#image-to-image) LCM demo as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i prompt="Astronauts in a jungle, cold color palette, muted colors, detailed, 8k" -i image=@astro.png

## Example:

"Astronauts in a jungle, cold color palette, muted colors, detailed, 8k"

![alt text](output.jpg)
