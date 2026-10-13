"""Render editable SVG and PNG diagrams for the AI Coding article (Pillow)."""
from pathlib import Path
from html import escape
import math
import os
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
FONTS = Path(os.environ.get('WINDIR', r'C:\Windows')) / 'Fonts'
BG, INK, MUTED = '#F5F3EC', '#20343D', '#5E7077'
TEAL, BLUE, RUST = '#087F78', '#4E7396', '#AD633F'

class Canvas:
    def __init__(self):
        self.image = Image.new('RGB', (1600, 1000), BG)
        self.draw = ImageDraw.Draw(self.image)
        self.elements = [f'<rect width="1600" height="1000" fill="{BG}"/>']

    def rect(self, box, fill, radius=18, stroke=None):
        self.draw.rounded_rectangle(box, radius=radius, fill=fill, outline=stroke, width=2)
        x, y, right, bottom = box
        border = f' stroke="{stroke}" stroke-width="2"' if stroke else ''
        self.elements.append(f'<rect x="{x}" y="{y}" width="{right-x}" height="{bottom-y}" rx="{radius}" fill="{fill}"{border}/>')

    def text(self, x, y, value, size=28, bold=False, color=INK, center=False, max_width=None):
        font = ImageFont.truetype(str(FONTS / ('msyhbd.ttc' if bold else 'msyh.ttc')), size)
        width = self.draw.textlength(value, font=font)
        if max_width is not None:
            assert width <= max_width, (value, width, max_width)
        if center:
            x -= width / 2
        self.draw.text((x, y), value, font=font, fill=color)
        # Match Pillow's visible glyph top and retain editable text in SVG.
        top = font.getbbox(value)[1]
        self.elements.append(f'<text x="{x:.2f}" y="{y+top:.2f}" dominant-baseline="hanging" font-family="Microsoft YaHei, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{escape(value)}</text>')

    def line(self, points, color=TEAL, width=4, arrow=False):
        self.draw.line(points, fill=color, width=width, joint='curve')
        self.elements.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"/>')
        if arrow:
            x, y = points[-1]
            a, b = points[-2]
            angle = math.atan2(y-b, x-a)
            head = [(x,y), (x-15*math.cos(angle)+7*math.sin(angle), y-15*math.sin(angle)-7*math.cos(angle)),
                    (x-15*math.cos(angle)-7*math.sin(angle), y-15*math.sin(angle)+7*math.cos(angle))]
            self.polygon(head, color)

    def polygon(self, points, fill):
        self.draw.polygon(points, fill=fill)
        self.elements.append(f'<polygon points="{" ".join(f"{x:.2f},{y:.2f}" for x,y in points)}" fill="{fill}"/>')

    def header(self, title, subtitle):
        self.rect((72,68,132,80),TEAL,6)
        self.text(72,110,title,50,True,max_width=1456)
        self.text(75,190,subtitle,25,color=MUTED,max_width=1450)

    def save(self, name):
        self.image.save(OUT / f'{name}.png')
        svg = '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">\n' + '\n'.join(self.elements) + '\n</svg>\n'
        (OUT / f'{name}.svg').write_text(svg, encoding='utf-8')


def delivery_loop():
    c = Canvas()
    c.header('一次任务的交付闭环', '人明确目标、判断边界并验收；AI 承担查阅、实现和执行检查。')
    c.rect((75,250,95,270),TEAL,5)
    c.text(108,242,'人：关键判断与责任',24,color=TEAL)
    c.rect((470,250,490,270),BLUE,5)
    c.text(503,242,'AI：执行与辅助检视',24,color=BLUE)
    stages = [
        ('目标与边界','先定义什么算完成','澄清与核查','目标与约束'),
        ('制定计划','方案、范围与影响','查代码、出方案','取舍与确认'),
        ('小步执行','实施受控的变更','实现与运行检查','观察与纠偏'),
        ('验证结果','用证据检查实际行为','检查与辅助检视','验收与风险判断'),
        ('交付与复用','让下一次少走弯路','整理变更与记录','交付与发布确认'),
    ]
    centers=[]
    for i, (title, detail, agent, human) in enumerate(stages):
        x = 80 + i*300
        centers.append(x+120)
        c.rect((x,310,x+240,590),'#E4EFED',stroke='#C8DBD6')
        c.text(x+20,333,f'0{i+1}',22,True,color=TEAL)
        c.text(x+120,378,title,30,True,center=True,max_width=210)
        c.text(x+120,438,detail,21,center=True,color=MUTED,max_width=210)
        c.text(x+20,493,'AI：'+agent,20,color=BLUE,max_width=210)
        c.text(x+20,541,'人：'+human,20,color=TEAL,max_width=210)
        if i<4:
            c.line([(x+248,404),(x+289,404)],arrow=True)
    # Evidence failure returns to the relevant part, not to more blind generation.
    c.rect((380,675,1220,754),'#E3EAF0',radius=16)
    c.text(800,696,'验证未通过：定位问题，回到计划或实现',27,True,center=True,max_width=790)
    c.line([(1100,590),(1100,675)],BLUE,arrow=True)
    c.line([(500,675),(500,601)],BLUE,arrow=True)
    c.line([(800,675),(800,601)],BLUE,arrow=True)
    # The learning/asset loop feeds the next task after delivery.
    c.line([(1400,590),(1400,819),(200,819),(200,601)],TEAL,arrow=True)
    c.rect((260,796,1320,850),BG,radius=0)
    c.text(790,804,'交付后沉淀约定、验证步骤与 skill，供下一次任务复用',26,True,color=TEAL,center=True,max_width=1050)
    c.text(75,925,'计划、执行、验证、交付都要完成；展开深度随任务的不确定性与影响调整。',25,color=MUTED,max_width=1450)
    c.save('task-delivery-loop')


def judgment_bottleneck():
    c = Canvas()
    c.header('生成快，不等于有效产出多', '候选内容必须经过理解与验证，才能成为可以负责的结果。')
    c.text(190,273,'AI 生成',29,True,center=True,color=BLUE)
    for i, value in enumerate(['代码与改动','测试与报告','方案与解释']):
        y=340+i*85
        c.rect((85,y,300,y+64),'#E3EAF0',radius=12)
        c.text(192,y+13,value,25,center=True,max_width=195)
    # Widths communicate a capacity mismatch; they are not measured rates.
    c.polygon([(365,320),(620,421),(620,539),(365,650)],'#D6E6E2')
    c.line([(315,450),(397,450)],BLUE,arrow=True)
    c.rect((620,421,975,539),TEAL,radius=0)
    c.text(797,429,'理解与验证',34,True,color='white',center=True,max_width=335)
    c.text(797,486,'关键边界 · 独立证据',22,color='white',center=True,max_width=335)
    c.text(797,338,'判别通道',26,True,color=TEAL,center=True)
    c.polygon([(975,421),(1150,380),(1150,580),(975,539)],'#D6E6E2')
    c.line([(1157,480),(1204,480)],arrow=True)
    c.rect((1210,370,1520,590),'#D0E5E1')
    c.text(1365,411,'有效产出',36,True,center=True,max_width=280)
    c.text(1365,481,'可理解 · 有证据 · 能负责',22,center=True,color=TEAL,max_width=285)
    c.text(85,630,'超出判别能力的候选内容',25,color=MUTED,max_width=460)
    c.line([(470,608),(470,692),(340,692),(340,729)],RUST,arrow=True)
    c.line([(850,550),(850,692),(1210,692),(1210,729)],RUST,arrow=True)
    c.rect((80,740,640,840),'#EFE5D8')
    c.text(110,754,'堆积：形成待检视队列',28,True,color=RUST,max_width=500)
    c.text(110,802,'交付等待，后续检视成本累积',22,color=MUTED,max_width=500)
    c.rect((900,740,1520,840),'#EFE5D8')
    c.text(930,754,'放水：把风险带进交付',28,True,color=RUST,max_width=560)
    c.text(930,802,'以“看起来正确”代替可靠证据',22,color=MUTED,max_width=560)
    c.rect((80,870,1520,948),'#E4EFED')
    c.text(800,888,'学习与练习 + 可靠的自动化验证 → 扩大判别通道',30,True,color=TEAL,center=True,max_width=1390)
    c.text(75,966,'概念示意：图形宽度不表示实际数量或比例。',18,color=MUTED)
    c.save('generation-judgment-bottleneck')

if __name__ == '__main__':
    delivery_loop()
    judgment_bottleneck()
