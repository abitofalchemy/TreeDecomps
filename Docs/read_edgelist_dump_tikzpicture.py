import sys

with open (sys.argv[1], 'r') as f:
    lines = f.readlines()
with open (sys.argv[1].split(".")[0]+".tex", "w") as out_f:
    out_f.write("\\begin{tikzpicture}\n")
    out_f.write("  \\graph [spring layout, nodes={vertex}, node distance=10mm]\n")
    out_f.write("  {\n")
    for l in lines:
        #l = l.split("{'Trade Value (US$)':")
        src, trg, wt = l.split()
        ol = src +"/{} --[line width="+ wt +"pt] " +trg +"/{},\n"
        out_f.write("\t"+ ol)
    out_f.write("  };\n")
    out_f.write("  \\node[align=center,font=\\bfseries] (title)\n")
    out_f.write("  at (current bounding box.north)\n")

    out_f.write("  {\\small \\textbf{(A)}};\n")


    out_f.write("\end{tikzpicture}\n")

