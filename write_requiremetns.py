import pkg_resources

# 列出所有想要的包
packages = ['torch', 'transformers', "deepspeed", "accelerate", "peft", "numpy", "jinja2", "flash-attn", "datasets",
            "modelscope", "pydantic", "bitsandbytes"]

# 获取每个包的版本并打印
with open('requirements.txt', 'w') as f:
    for package in packages:
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"{package}=={version}")
            f.write(f"{package}=={version}\n")
        except Exception as e:
            continue
