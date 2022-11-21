from setuptools import setup, find_packages

def main():
    setup(
        name='bmcook',
        version='0.1.1.1',
        description="Model Compression for Big Models",
        author="Baitao Gong",
        author_email="gongbaitao11@gmail.com",
        packages=find_packages(exclude='cpm_live'),
        url="https://github.com/OpenBMB/BMCook",
        install_requires=[
            "bmtrain",
            "model_center"
        ],
        license="Apache 2.0"
    )

if __name__ == "__main__":
    main()