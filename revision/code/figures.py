"""Reproduce the six figures changed or added in the revision."""
from pathlib import Path
import json,math,re,runpy
from pypdf import PdfReader,PdfWriter
from pypdf.generic import ContentStream,DecodedStreamObject,NameObject
from paths import DATA,output,load

def write_stream(source,target,transform):
    reader=PdfReader(source);writer=PdfWriter();writer.clone_document_from_reader(reader)
    page=writer.pages[0];raw=page.get_contents().get_data()
    stream=DecodedStreamObject();stream.set_data(transform(raw))
    page[NameObject('/Contents')]=writer._add_object(stream);writer.write(target)

def figure2():
    def transform(raw):
        pattern=rb'q 69\.7 43\.73125 388\.8 277\.2 re W n /A3 gs 2 J 1 j 2 w 0 G /DeviceRGB cs\s+69\.7 273\.324766 m.*?\nS\nQ'
        matches=list(re.finditer(pattern,raw,re.S));assert len(matches)==1
        block=matches[0].group();edited=raw.replace(block,b'',1)
        legend=b'384.74 310.0655 m\n394.74 310.0655 l\n404.74 310.0655 l\n\nS'
        assert edited.count(legend)==1
        edited=edited.replace(legend,b'[6 4] 0 d\n'+legend+b'\n[] 0 d',1)
        marker=b'q 69.7 43.73125 388.8 277.2 re W n /A3 gs 0.1215686275 0.4666666667\n0.7058823529 rg'
        assert edited.count(marker)==1
        block=block.replace(b'/DeviceRGB cs',b'/DeviceRGB cs\n[6 4] 0 d',1)
        return edited.replace(marker,block+b'\n'+marker,1)
    write_stream(DATA/'source_figures/fig2_original.pdf',output('figures')/'fig2_static.pdf',transform)

def figure4():
    values=load('figure_data.json')['original_plot_values']['linear']
    rr=[r for r in values if r['J']==20]
    xs=[math.log10(r['C_plot']) for r in rr];ys=[118.736596,100.6821711,90.1599477,73.0505720,61.9900011]
    xm=sum(xs)/len(xs);ym=sum(ys)/len(ys)
    scale=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sum((x-xm)**2 for x in xs)
    row=next(x['adopted'] for x in load('replacement_points.json')['points'] if x['adopted']['model']=='linear')
    delta=scale*math.log10(row['new_refined_C']/row['original_plotted_C'])
    source=DATA/'source_figures/fig4_original.pdf'
    reader=PdfReader(source);ops=ContentStream(reader.pages[0].get_contents(),reader).operations
    targets=[]
    for index,expected,change in [(1514,-10.5222234,delta),(1516,-17.1093757,-delta)]:
        args,op=ops[index];assert abs(float(args[5])-expected)<1e-5
        targets.append((op,tuple(float(x) for x in args),5,change))
    def transform(raw):
        edits=[];number=rb'[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)'
        for op,args,position,change in targets:
            pattern=rb'(?<!\S)((?:'+number+rb'\s+){'+str(len(args)).encode()+rb'})'+op+rb'(?=\s|$)'
            matches=0
            for m in re.finditer(pattern,raw):
                tokens=list(re.finditer(number,m.group(1)));values=[float(x.group()) for x in tokens]
                if len(values)!=len(args) or any(abs(x-y)>1e-10 for x,y in zip(values,args)):continue
                token=tokens[position];new=format(values[position]+change,'.12f').rstrip('0').rstrip('.').encode()
                edits.append((m.start(1)+token.start(),m.start(1)+token.end(),new));matches+=1
            assert matches==1
        for start,end,new in sorted(edits,reverse=True):raw=raw[:start]+new+raw[end:]
        return raw
    write_stream(source,output('figures')/'fig4_linear.pdf',transform)

def make_figures():
    here=Path(__file__).resolve().parent;out=output('figures')
    figure2();figure4()
    runpy.run_path(str(here/'figure3.py'),run_name='__main__')
    runpy.run_path(str(here/'figures_extra.py'),run_name='__main__')
    for old,new in [('q3_budgets','fig6_initialization'),('q3_paths','figS2_coupling_steps'),('parameter_mismatch','figS1_parameter_mismatch')]:
        (out/(old+'.pdf')).replace(out/(new+'.pdf'))
    print('Rebuilt figures in',out)
