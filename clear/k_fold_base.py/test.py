from peft import LoraConfig, get_peft_model
from model_selector import ModelSelector
config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
model_selector = ModelSelector(
            model_type= 'timm',
            num_classes = 500,
            model_name= 'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',
            pretrained= True
        )
model = model_selector.get_model()
print(model)
# model = get_peft_model(model, config)
# model.print_trainable_parameters()