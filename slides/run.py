import subprocess


files = [
    "slide_01.py",
    #"slide_02.py",
    #"slide_03.py",
    #"slide_04.py",
    #"slide_05.py",
    #"slide_06.py",
    #"slide_07.py",
    #"slide_08.py",
    #"slide_09.py",
    #"slide_10.py",
    #"slide_11.py",
    #"slide_12.py",
    #"slide_13.py",
    #"slide_14.py",
    #"slide_15.py",
]



for f in files:
    print(f"\nProcessando: {f}")
    comando = ["manim", "-qh"  ,"-v", "WARNING"    , f, "generate"   ]
    subprocess.run(comando)
