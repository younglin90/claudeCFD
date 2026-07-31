#!/bin/bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
pandoc paper_jcp.md -o /tmp/paper_raw.docx --from markdown+tex_math_dollars+pipe_tables --standalone 2>&1 | head -5
python3 - <<'EOF'
import zipfile, shutil, re
src, dst = '/tmp/paper_raw.docx', 'paper_jcp_draft.docx'
shutil.copy(src, dst)
zin = zipfile.ZipFile(src)
items = zin.namelist()
# pandoc uses THEME fonts: patch theme1.xml typefaces (major=headings, minor=body)
theme = zin.read('word/theme/theme1.xml').decode('utf-8')
theme = re.sub(r'typeface="Calibri Light"', 'typeface="Times New Roman"', theme)
theme = re.sub(r'typeface="Calibri"', 'typeface="Times New Roman"', theme)
styles = zin.read('word/styles.xml').decode('utf-8')
# belt-and-braces: also force explicit run fonts in docDefaults
styles = re.sub(r'(<w:rPrDefault>\s*<w:rPr>)',
                r'\1<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>',
                styles, count=1)
styles = styles.replace('Calibri Light', 'Times New Roman').replace('"Calibri"', '"Times New Roman"')
# headings black (pandoc default is blue accent; attribute order varies) -> all explicit colors black
styles = re.sub(r'<w:color [^>]*/>', '<w:color w:val="000000"/>', styles)
# body 11pt -> 10pt (w:sz is half-points: 22 -> 20)
styles = styles.replace('<w:sz w:val="22"/>', '<w:sz w:val="20"/>').replace('<w:szCs w:val="22"/>', '<w:szCs w:val="20"/>')
# headings: H1 16pt->12pt bold-ish, H2 13pt->11pt, H3 12pt->10pt
styles = styles.replace('<w:sz w:val="32"/>', '<w:sz w:val="24"/>').replace('<w:szCs w:val="32"/>', '<w:szCs w:val="24"/>')
styles = styles.replace('<w:sz w:val="26"/>', '<w:sz w:val="22"/>').replace('<w:szCs w:val="26"/>', '<w:szCs w:val="22"/>')
styles = styles.replace('<w:sz w:val="24"/>', '<w:sz w:val="20"/>').replace('<w:szCs w:val="24"/>', '<w:szCs w:val="20"/>')
# paragraph gaps 50% (page budget; halves pandoc before/after spacing, line spacing untouched)
styles = re.sub(r'<w:spacing[^/]*/>',
                lambda m: re.sub(r'w:(after|before)="(\d+)"',
                                 lambda x: 'w:%s="%d"' % (x.group(1), int(x.group(2)) // 2), m.group(0)),
                styles)
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zout:
    for it in items:
        if it == 'word/styles.xml': data = styles.encode('utf-8')
        elif it == 'word/theme/theme1.xml': data = theme.encode('utf-8')
        else: data = zin.read(it)
        zout.writestr(it, data)
print('patched -> paper_jcp_draft.docx')
EOF
# stats
python3 - <<'EOF'
import zipfile, re
z = zipfile.ZipFile('paper_jcp_draft.docx')
doc = z.read('word/document.xml').decode('utf-8')
words = len(re.sub(r'<[^>]+>', ' ', doc).split())
eqs = doc.count('<m:oMath')
tbls = doc.count('<w:tbl>')
print('words=%d equations=%d tables=%d' % (words, eqs, tbls))
st = z.read('word/styles.xml').decode('utf-8')
th = z.read('word/theme/theme1.xml').decode('utf-8')
print('TNR styles:', st.count('Times New Roman'), '| TNR theme:', th.count('Times New Roman'), '| sz20:', st.count('w:sz w:val="20"'))
EOF
cp paper_jcp_draft.docx /mnt/c/Users/user/Downloads/paper_jcp_draft.docx && echo COPIED
