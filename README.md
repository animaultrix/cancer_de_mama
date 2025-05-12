# Clonar y ejecutar

    python -m venv .venv
    .venv\Scripts\activate  

# Instalar dependencias

    pip install -r requirements.txt

# Ejecutar test.py para ver si se esta usando GPU (TensorFlow use CUDA)

    phyton test.py

    - Si no se esta usando la gpu se puede configurar con:

        Instala Anaconda  👉  https://docs.conda.io/en/latest/miniconda.html

        ✅ Elige la versión de Windows x86_64 - Python 3.10

        Durante la instalación:

        Marca la opción “Add Miniconda to my PATH environment variable” (aunque diga “no recomendado”).
        Abre el menú Inicio (Windows)

        Abre el menú Inicio (Windows) escribe y abre: Anaconda Prompt

        Ejecuta en orden:
        
            conda create -n tf-gpu python=3.10

            conda activate tf-gpu

            pip install tensorflow==2.11.0
