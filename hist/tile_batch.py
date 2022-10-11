if __name__ == "__main__":
    import pandas as pd
    
    df = pd.read_csv("metaTR.csv")
    slides = df.loc[df["tiff.XResolution"]==50000, "filename"].to_list()
    with open("slides.txt", "w") as f:
        f.write("\n".join(slides))

