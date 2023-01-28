import numpy as np
from matplotlib import pyplot as plt
import torch
from diffusers import StableDiffusionPipeline


class GeneradorImagenes():
    
    def __init__(self):
        self.pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)  
        self.pipe = self.pipe.to("cuda")
        self.generator = torch.Generator("cuda").manual_seed(np.random.randint(2048))


    def GenerarImagen(self,
                prompt, 
                height=512, 
                width=512, 
                num_inference_steps=30,
                negative_prompt='cartoon',
                num_images_per_prompt=1,
                guidance_scale = 5.0,
                output_type = 'np.array'):


        imagenes = self.pipe(prompt, 
                            height=height, 
                            width=width, 
                            num_inference_steps=num_inference_steps,
                            negative_prompt=negative_prompt,
                            num_images_per_prompt=num_images_per_prompt,
                            guidance_scale = guidance_scale,
                            output_type = output_type,
                            generator=self.generator).images

        if (num_images_per_prompt==1): imagenes = imagenes[0]
        
        return imagenes