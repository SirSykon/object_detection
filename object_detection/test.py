
import yaml

yaml_file = open("config.yml", 'r')
print(yaml_file)
yaml_content = yaml.load(yaml_file, Loader=yaml.FullLoader)

print(yaml_content)
print("Key: Value")
for key, value in yaml_content.items():
    print(f"{key}: {value}")