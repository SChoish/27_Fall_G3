"""IEEE transistor schematics that match the ADC and D2D netlists."""
from pathlib import Path
from PIL import Image
import schemdraw
from schemdraw import elements as elm

ROOT = Path(__file__).resolve().parents[1]


def nfet():
    return elm.AnalogNFet(bulk=True, source_arrow=True).reverse()


def pfet():
    return elm.AnalogPFet(bulk=True, source_arrow=True).reverse()


def rail(d, x0, x1, y, name):
    d.add(elm.Line().at((x0, y)).to((x1, y)).linewidth(1.8))
    d.add(elm.Label().at((x0 - 0.15, y + 0.55)).label(name, loc='right'))


def wire(d, a, b, shape='-'):
    d.add(elm.Wire(shape).at(a).to(b))


def note(d, x, y, text, loc='right'):
    d.add(elm.Label().at((x, y)).label(text, loc=loc))


def save(d, path):
    path = Path(path).with_suffix('.jpg')
    tmp = path.with_suffix('.png')
    d.draw(show=False)
    d.save(str(tmp), dpi=220)
    png = Image.open(tmp).convert('RGBA')
    jpg = Image.new('RGB', png.size, (255, 255, 255))
    jpg.paste(png, mask=png.split()[-1])
    jpg.save(path, 'JPEG', quality=95, optimize=True, subsampling=0)
    tmp.unlink(missing_ok=True)
    print(path)


def draw_adc(path):
    d = schemdraw.Drawing(unit=2.4, fontsize=9, lw=1.15, inches_per_unit=0.55)
    d.config(font='DejaVu Sans')

    yv = 20.0
    rail(d, 0.4, 16.0, yv, 'AVDD  1.8 V')

    iref = d.add(elm.SourceI().down().at((1.35, yv)))
    note(d, 0.05, (yv + iref.end.y) / 2, 'IREF')
    ref = d.add(nfet().anchor('drain').at((3.35, iref.end.y)))
    wire(d, iref.end, ref.drain)
    d.add(elm.Dot().at(ref.drain))
    note(d, ref.drain.x + 0.45, ref.drain.y + 0.40, 'bias')
    note(d, ref.center.x + 0.60, ref.center.y - 0.15, 'REF')
    wire(d, ref.gate, ref.drain, '|-')
    d.add(elm.Ground().at(ref.source))

    p1 = d.add(pfet().anchor('source').at((6.4, yv)))
    p2 = d.add(pfet().anchor('source').at((10.0, yv)))
    p3 = d.add(pfet().anchor('source').at((13.8, yv)))
    for p, name in ((p1, 'P1'), (p2, 'P2'), (p3, 'P3')):
        d.add(elm.Dot().at(p.source))
        note(d, p.center.x + 0.55, p.center.y + 0.15, name)

    gap = 1.6
    in1 = d.add(nfet().anchor('drain').at((6.4, p1.drain.y - gap)))
    in2 = d.add(nfet().anchor('drain').at((10.0, p2.drain.y - gap)))
    wire(d, p1.drain, in1.drain)
    wire(d, p2.drain, in2.drain)
    d.add(elm.Dot().at(in1.drain))
    d.add(elm.Dot().at(in2.drain))
    note(d, in1.drain.x - 0.55, in1.drain.y, 'n1')
    note(d, in2.drain.x + 0.35, in2.drain.y + 0.35, 'n2')
    note(d, in1.center.x + 0.55, in1.center.y, 'IN1')
    note(d, in2.center.x + 0.55, in2.center.y, 'IN2')

    wire(d, in1.source, in2.source)
    tx = (in1.source.x + in2.source.x) / 2
    d.add(elm.Dot().at((tx, in1.source.y)))
    note(d, tx - 0.70, in1.source.y + 0.20, 'tail')
    tail = d.add(nfet().anchor('drain').at((tx, in1.source.y)))
    note(d, tail.center.x + 0.65, tail.center.y - 0.10, 'TAIL')
    d.add(elm.Ground().at(tail.source))

    wire(d, p1.gate, p1.drain, '|-')
    wire(d, p2.gate, (p1.gate.x, p2.gate.y))
    d.add(elm.Dot().at((p1.gate.x, p1.gate.y)))

    n3 = d.add(nfet().anchor('drain').at((13.8, p3.drain.y - gap)))
    wire(d, p3.drain, n3.drain)
    d.add(elm.Ground().at(n3.source))
    note(d, n3.center.x + 0.55, n3.center.y, 'N3')
    out = (13.8, (p3.drain.y + n3.drain.y) / 2)
    d.add(elm.Dot().at(out))
    note(d, out[0] + 0.15, out[1] + 0.45, 'out')

    wire(d, p3.gate, (p3.gate.x, p2.drain.y))
    wire(d, (p3.gate.x, p2.drain.y), p2.drain)

    wire(d, tail.gate, (ref.drain.x, tail.gate.y))
    d.add(elm.Dot())
    wire(d, n3.gate, (ref.drain.x, n3.gate.y))
    wire(d, (ref.drain.x, tail.gate.y), (ref.drain.x, n3.gate.y))
    wire(d, ref.drain, (ref.drain.x, n3.gate.y))
    d.add(elm.Dot().at((ref.drain.x, n3.gate.y)))
    d.add(elm.Dot().at((ref.drain.x, tail.gate.y)))

    d.add(elm.Capacitor().endpoints(in2.drain, out))
    note(d, (in2.drain.x + out[0]) / 2, (in2.drain.y + out[1]) / 2 - 0.70, 'CC  0.3–6 pF')

    out2 = (16.6, out[1])
    wire(d, out, out2)
    d.add(elm.Dot().at(out2))
    d.add(elm.Capacitor().down().at(out2))
    d.add(elm.Ground())
    note(d, out2[0] + 0.85, out2[1] - 1.15, 'COUT  2 pF')

    fb_y = tail.source.y - 2.0
    wire(d, out2, (out2[0], fb_y))
    wire(d, (out2[0], fb_y), (in1.gate.x, fb_y))
    wire(d, (in1.gate.x, fb_y), in1.gate)
    note(d, (out2[0] + in1.gate.x) / 2, fb_y + 0.40, 'inn = out')

    vin_y = fb_y - 1.7
    wire(d, (0.2, vin_y), (in2.gate.x, vin_y))
    wire(d, (in2.gate.x, vin_y), in2.gate)
    d.add(elm.Dot().at((0.2, vin_y)))
    note(d, -0.15, vin_y + 0.40, 'VIN = inp')
    note(d, 10.4, vin_y - 0.50, 'XAMP  vin out out avdd')

    riso = d.add(elm.Resistor().right().length(3.8).at(out2))
    note(d, (out2[0] + riso.end.x) / 2, out2[1] + 0.55, 'RISO  20–2 kΩ')
    adc = riso.end
    d.add(elm.Dot().at(adc))
    note(d, adc.x, adc.y + 0.55, 'adc_in')
    d.add(elm.Capacitor().down().at(adc))
    d.add(elm.Ground())
    note(d, adc.x + 0.95, adc.y - 1.15, 'CFILTER  1–100 pF')

    swp = d.add(elm.AnalogPFet(bulk=True, source_arrow=True).theta(-90)
                .at((adc.x + 2.8, adc.y + 1.55)))
    swn = d.add(elm.AnalogNFet(bulk=True, source_arrow=True).theta(-90)
                .at((adc.x + 2.8, adc.y - 1.55)))
    note(d, swp.center.x, swp.center.y + 0.85, 'SWP')
    note(d, swn.center.x, swn.center.y - 0.85, 'SWN')
    wire(d, adc, (swp.source.x, adc.y))
    wire(d, (swp.source.x, adc.y), swp.source)
    wire(d, (swn.source.x, adc.y), swn.source)
    d.add(elm.Dot().at((swp.source.x, adc.y)))
    wire(d, swp.drain, (swp.drain.x, adc.y))
    wire(d, swn.drain, (swn.drain.x, adc.y))
    d.add(elm.Dot().at((swp.drain.x, adc.y)))
    samp = (swp.drain.x + 2.4, adc.y)
    wire(d, (swp.drain.x, adc.y), samp)
    d.add(elm.Dot().at(samp))
    note(d, samp[0], samp[1] + 0.55, 'sample')
    d.add(elm.Capacitor().down().at(samp))
    d.add(elm.Ground())
    note(d, samp[0] + 1.05, samp[1] - 1.15, 'CSAMPLE  15 pF')

    yc = vin_y - 3.2
    rail(d, 0.4, 10.2, yc, 'CLKDD  1.8 V')
    mid = yc - 2.35
    d.add(elm.Dot().at((0.2, mid)))
    note(d, -0.15, mid + 0.40, 'clkcmd')
    wire(d, (0.2, mid), (2.15, mid))

    ck0p = d.add(pfet().anchor('source').at((3.3, yc)))
    ck0n = d.add(nfet().anchor('source').at((3.3, yc - 4.7)))
    note(d, ck0p.center.x + 0.55, ck0p.center.y, 'CK0P')
    note(d, ck0n.center.x + 0.55, ck0n.center.y, 'CK0N')
    d.add(elm.Ground().at(ck0n.source))
    wire(d, ck0p.drain, ck0n.drain)
    d.add(elm.Dot().at((3.3, mid)))
    note(d, 3.3, mid + 0.40, 'clkp')
    wire(d, ck0p.gate, (2.15, ck0p.gate.y))
    wire(d, (2.15, ck0p.gate.y), (2.15, mid))
    wire(d, ck0n.gate, (2.15, ck0n.gate.y))
    wire(d, (2.15, ck0n.gate.y), (2.15, mid))
    d.add(elm.Dot().at((2.15, mid)))

    wire(d, (3.3, mid), (5.15, mid))
    ck1p = d.add(pfet().anchor('source').at((6.3, yc)))
    ck1n = d.add(nfet().anchor('source').at((6.3, yc - 4.7)))
    note(d, ck1p.center.x + 0.55, ck1p.center.y, 'CK1P')
    note(d, ck1n.center.x + 0.55, ck1n.center.y, 'CK1N')
    d.add(elm.Ground().at(ck1n.source))
    wire(d, ck1p.drain, ck1n.drain)
    d.add(elm.Dot().at((6.3, mid)))
    note(d, 6.3, mid + 0.40, 'clkn')
    wire(d, ck1p.gate, (5.15, ck1p.gate.y))
    wire(d, (5.15, ck1p.gate.y), (5.15, mid))
    wire(d, ck1n.gate, (5.15, ck1n.gate.y))
    wire(d, (5.15, ck1n.gate.y), (5.15, mid))
    d.add(elm.Dot().at((5.15, mid)))

    wire(d, (6.3, mid), (swn.gate.x, mid))
    wire(d, (swn.gate.x, mid), swn.gate)
    bot = ck0n.source.y - 1.2
    wire(d, (5.15, mid), (5.15, bot))
    wire(d, (5.15, bot), (swp.gate.x, bot))
    wire(d, (swp.gate.x, bot), swp.gate)

    note(d, 0.2, bot - 1.3,
         'Unity-gain follower:  inp = vin,  inn = out.   '
         'Complementary TG clocks from two CMOS inverters.   RLEAK 1 TΩ omitted.')
    save(d, path)


def inverter(d, x, yv, ybot, gx, names, out_name):
    p = d.add(pfet().anchor('source').at((x, yv)))
    n = d.add(nfet().anchor('source').at((x, ybot)))
    note(d, p.center.x + 0.15, yv - 0.28, names[0])
    note(d, n.center.x + 0.15, ybot + 0.28, names[1])
    d.add(elm.Ground().at(n.source))
    wire(d, p.drain, n.drain)
    midy = (p.drain.y + n.drain.y) / 2
    d.add(elm.Dot().at((x, midy)))
    note(d, x + 0.55, midy + 0.28, out_name)
    wire(d, p.gate, (gx, p.gate.y))
    wire(d, (gx, p.gate.y), (gx, midy))
    wire(d, n.gate, (gx, n.gate.y))
    wire(d, (gx, n.gate.y), (gx, midy))
    d.add(elm.Dot().at((gx, midy)))
    return (x, midy)


def draw_d2d(path):
    d = schemdraw.Drawing(unit=2.15, fontsize=9, lw=1.15, inches_per_unit=0.50)
    d.config(font='DejaVu Sans')

    def cell_and_pre(tag, yv):
        inv_bot = yv - 5.0
        gnd_y = yv - 7.4
        ysig = (yv + inv_bot) / 2
        rail(d, 0.2, 9.4, yv, f'VDD  1.0 V     lane {tag}')
        wire(d, (0.2, ysig), (1.25, ysig))
        note(d, 0.05, ysig + 0.45, f'din{tag}')
        pre = inverter(d, 2.35, yv, inv_bot, 1.25, ('MPRE1P', 'MPRE1N'), f'pre{tag}')
        wire(d, pre, (3.7, pre[1]))
        drive = inverter(d, 4.85, yv, inv_bot, 3.7, ('MPRE2P', 'MPRE2N'), f'drive{tag}')
        wire(d, drive, (6.2, drive[1]))

        mh = d.add(pfet().anchor('source').at((7.15, yv)))
        note(d, mh.center.x - 0.95, mh.center.y - 0.20, f'MH{tag}')
        d.add(elm.Dot().at(mh.source))
        mp = d.add(pfet().anchor('source').at((8.85, mh.drain.y)))
        note(d, mp.center.x + 0.55, mp.center.y, f'MP{tag}')
        wire(d, mh.drain, mp.source)
        d.add(elm.Dot().at(mp.source))
        note(d, mp.source.x - 0.15, mp.source.y - 0.40, f'hp{tag}')
        d.add(elm.Line().at(mh.gate).left(1.15))
        note(d, mh.gate.x - 1.55, yv + 0.55, f'enb{tag}')

        mf = d.add(nfet().anchor('source').at((7.15, gnd_y)))
        note(d, mf.center.x - 0.85, mf.center.y, f'MF{tag}')
        d.add(elm.Ground().at(mf.source))
        mn = d.add(nfet().anchor('source').at((8.85, mf.drain.y)))
        note(d, mn.center.x + 0.55, mn.center.y, f'MN{tag}')
        wire(d, mf.drain, mn.source)
        d.add(elm.Dot().at(mn.source))
        note(d, mn.source.x - 0.15, mn.source.y + 0.40, f'fn{tag}')
        d.add(elm.Line().at(mf.gate).left(1.15))
        note(d, mf.gate.x - 1.45, gnd_y - 0.85, f'en{tag}')

        wire(d, mp.drain, mn.drain)
        txy = (mp.drain.y + mn.drain.y) / 2
        d.add(elm.Dot().at((8.85, txy)))
        note(d, 8.20, txy + 0.35, f'tx{tag}')
        wire(d, mp.gate, (mp.gate.x, drive[1]))
        wire(d, (mp.gate.x, drive[1]), (6.2, drive[1]))
        wire(d, mn.gate, (mn.gate.x, drive[1]))
        wire(d, (8.85, txy), (10.5, txy))
        note(d, 9.55, yv + 0.55, '×8')
        return (10.5, txy), mf.source.y

    (tx0, g0) = cell_and_pre('0', 28.0)
    (tx1, g1) = cell_and_pre('1', 13.2)

    def rlc(start, tag):
        here = start
        nodes = []
        for i, nm in enumerate(('ch0', 'ch1', 'rx')):
            r = d.add(elm.Resistor().right().length(3.1).at(here))
            note(d, (here[0] + r.end.x) / 2, here[1] + 0.60, f'R{i}')
            ind = d.add(elm.Inductor().right().length(3.1).at(r.end))
            note(d, (r.end.x + ind.end.x) / 2, here[1] - 0.55, f'L{i}')
            d.add(elm.Dot().at(ind.end))
            note(d, ind.end.x, ind.end.y + 0.72, f'{nm}_{tag}')
            d.add(elm.Capacitor().down().at(ind.end))
            d.add(elm.Ground())
            note(d, ind.end.x + 0.80, ind.end.y - 1.25, f'C{i}')
            nodes.append(ind.end)
            gap = d.add(elm.Line().at(ind.end).right(0.55))
            here = (gap.end.x, gap.end.y)
        crx = d.add(elm.Line().at(here).right(2.2))
        d.add(elm.Capacitor().down().at(crx.end))
        d.add(elm.Ground())
        note(d, crx.end.x + 0.80, crx.end.y - 1.20, 'CRX')
        return nodes

    n0 = rlc(tx0, '0')
    n1 = rlc(tx1, '1')
    for a, b in zip(n0, n1):
        d.add(elm.Capacitor().endpoints(a, b))
        note(d, a.x + 0.55, (a.y + b.y) / 2, 'Cc/3')

    note(d, 0.2, 2.0, 'Two CMOS predrivers + 8 header/footer cells per lane.')
    note(d, 0.2, 1.35, 'Disabled cells stay in the netlist.  RX is behavioral Python.')
    save(d, path)


def pin(el, name):
    return getattr(el, name)


def cell(d, title, pins, at, anchor='center', size=(3.2, 2.5)):
    ic = d.add(elm.Ic(size=list(size), pins=pins).anchor(anchor).at(at))
    d.add(elm.Label().at(ic.center).label(title, fontsize=13))
    return ic


def ipin(name, side, pos=None):
    return elm.IcPin(name='', side=side, pos=pos, anchorname=name)


def draw_adc_cells(path):
    """Hierarchical ADC driver: AMP / INV / TG / ADC cells plus passives."""
    d = schemdraw.Drawing(unit=2.2, fontsize=10, lw=1.2, inches_per_unit=0.55)
    d.config(font='DejaVu Sans')

    yv = 13.4
    rail(d, 0.3, 24.0, yv, 'AVDD  1.8 V')

    amp = cell(d, 'AMP', [
        ipin('inn', 'left'),
        ipin('inp', 'left'),
        ipin('out', 'right'),
        ipin('vdd', 'top'),
    ], at=(7.0, 10.3), anchor='center', size=(3.6, 3.0))
    wire(d, amp.vdd, (amp.vdd.x, yv))
    d.add(elm.Dot().at((amp.vdd.x, yv)))
    note(d, amp.inp.x - 0.15, amp.inp.y + 0.40, 'inp')
    note(d, amp.inn.x - 0.15, amp.inn.y - 0.40, 'inn')
    note(d, amp.out.x + 0.15, amp.out.y + 0.40, 'out')

    d.add(elm.Dot().at((0.3, amp.inp.y)))
    note(d, 0.10, amp.inp.y + 0.45, 'VIN')
    wire(d, (0.3, amp.inp.y), amp.inp)

    out = (amp.out.x + 1.8, amp.out.y)
    wire(d, amp.out, out)
    d.add(elm.Dot().at(out))
    note(d, out[0], out[1] + 0.50, 'out')
    d.add(elm.Capacitor().down().at(out))
    d.add(elm.Ground())
    note(d, out[0] + 0.95, out[1] - 1.20, 'COUT  2 pF')

    wire(d, out, (out[0], amp.inn.y))
    wire(d, (out[0], amp.inn.y), amp.inn)
    d.add(elm.Dot().at((out[0], amp.inn.y)))
    note(d, (amp.inn.x + out[0]) / 2 - 0.2, amp.inn.y - 0.45, 'inn = out')

    riso = d.add(elm.Resistor().right().length(3.8).at(out))
    note(d, (out[0] + riso.end.x) / 2, out[1] + 0.55, 'RISO  20–2 kΩ')
    adc_in = riso.end
    d.add(elm.Dot().at(adc_in))
    note(d, adc_in.x, adc_in.y + 0.55, 'adc_in')
    d.add(elm.Capacitor().down().at(adc_in))
    d.add(elm.Ground())
    note(d, adc_in.x + 1.10, adc_in.y - 1.20, 'CFILTER  1–100 pF')

    tg = cell(d, 'TG', [
        ipin('ain', 'left', 1.25),
        ipin('aout', 'right', 1.25),
        ipin('clkp', 'bottom', 0.85),
        ipin('clkn', 'bottom', 2.15),
    ], at=(adc_in.x + 2.6, adc_in.y), anchor='ain', size=(3.0, 2.6))
    wire(d, adc_in, tg.ain)
    note(d, tg.clkp.x - 0.15, tg.clkp.y - 0.40, 'clkp')
    note(d, tg.clkn.x + 0.15, tg.clkn.y - 0.40, 'clkn')

    samp = (tg.aout.x + 2.0, tg.aout.y)
    wire(d, tg.aout, samp)
    d.add(elm.Dot().at(samp))
    note(d, samp[0], samp[1] + 0.50, 'sample')
    d.add(elm.Capacitor().down().at(samp))
    d.add(elm.Ground())
    note(d, samp[0] + 1.10, samp[1] - 1.20, 'CSAMPLE  15 pF')

    adc = cell(d, 'ADC', [
        ipin('ain', 'left', 1.25),
        ipin('vdd', 'top'),
    ], at=(samp[0] + 2.4, samp[1]), anchor='ain', size=(2.7, 2.4))
    wire(d, samp, adc.ain)
    wire(d, adc.vdd, (adc.vdd.x, yv))
    d.add(elm.Dot().at((adc.vdd.x, yv)))

    yc = 5.6
    rail(d, 0.3, 11.2, yc, 'CLKDD  1.8 V')
    inv0 = cell(d, 'INV', [
        ipin('ain', 'left', 1.1),
        ipin('aout', 'right', 1.1),
        ipin('vdd', 'top'),
    ], at=(4.8, yc), anchor='vdd', size=(2.6, 2.1))
    inv1 = cell(d, 'INV', [
        ipin('ain', 'left', 1.1),
        ipin('aout', 'right', 1.1),
        ipin('vdd', 'top'),
    ], at=(9.0, yc), anchor='vdd', size=(2.6, 2.1))
    wire(d, inv0.vdd, (inv0.vdd.x, yc))
    wire(d, inv1.vdd, (inv1.vdd.x, yc))
    d.add(elm.Dot().at((inv0.vdd.x, yc)))
    d.add(elm.Dot().at((inv1.vdd.x, yc)))

    d.add(elm.Dot().at((0.3, inv0.ain.y)))
    note(d, 0.10, inv0.ain.y + 0.45, 'clkcmd')
    wire(d, (0.3, inv0.ain.y), inv0.ain)
    wire(d, inv0.aout, inv1.ain)
    d.add(elm.Dot().at(inv0.aout))
    note(d, inv0.aout.x, inv0.aout.y + 0.50, 'clkp')
    note(d, inv1.aout.x + 0.40, inv1.aout.y + 0.50, 'clkn')

    bot_p, bot_n = 2.6, 1.9
    wire(d, inv0.aout, (inv0.aout.x, bot_p))
    wire(d, (inv0.aout.x, bot_p), (tg.clkp.x, bot_p))
    wire(d, (tg.clkp.x, bot_p), tg.clkp)
    wire(d, inv1.aout, (inv1.aout.x, bot_n))
    wire(d, (inv1.aout.x, bot_n), (tg.clkn.x, bot_n))
    wire(d, (tg.clkn.x, bot_n), tg.clkn)

    note(d, 0.3, 0.85, 'AMP = 8-MOS Miller opamp (XAMP vin out out avdd).   TG = SWN/SWP.   ADC = sampler load.')
    save(d, path)


def draw_d2d_cells(path):
    """Hierarchical D2D: TX cell, RLC channel, RX cell."""
    d = schemdraw.Drawing(unit=2.15, fontsize=10, lw=1.2, inches_per_unit=0.52)
    d.config(font='DejaVu Sans')

    def lane(tag, y_rail, y_sig):
        rail(d, 0.3, 6.4, y_rail, f'VDD  1.0 V    lane {tag}')
        tx = cell(d, 'TX  ×8', [
            ipin('din', 'left', 1.25),
            ipin('tx', 'right', 1.25),
            ipin('vdd', 'top'),
            ipin('en', 'bottom', 1.25),
        ], at=(4.4, y_sig), anchor='center', size=(3.2, 2.5))
        wire(d, tx.vdd, (tx.vdd.x, y_rail))
        d.add(elm.Dot().at((tx.vdd.x, y_rail)))
        d.add(elm.Dot().at((0.3, tx.din.y)))
        note(d, 0.10, tx.din.y + 0.45, f'din{tag}')
        wire(d, (0.3, tx.din.y), tx.din)
        note(d, tx.en.x, tx.en.y - 0.45, f'en{tag}')

        here = (tx.tx.x + 1.0, tx.tx.y)
        wire(d, tx.tx, here)
        d.add(elm.Dot().at(here))
        note(d, here[0], here[1] + 0.45, f'tx{tag}')
        nodes = []
        for i, nm in enumerate(('ch0', 'ch1', 'rx')):
            r = d.add(elm.Resistor().right().length(3.0).at(here))
            note(d, (here[0] + r.end.x) / 2, here[1] + 0.55, f'R{i}')
            ind = d.add(elm.Inductor().right().length(3.0).at(r.end))
            note(d, (r.end.x + ind.end.x) / 2, here[1] - 0.50, f'L{i}')
            d.add(elm.Dot().at(ind.end))
            note(d, ind.end.x, ind.end.y + 0.55, f'{nm}_{tag}')
            d.add(elm.Capacitor().down().at(ind.end))
            d.add(elm.Ground())
            note(d, ind.end.x + 0.70, ind.end.y - 1.15, f'C{i}')
            nodes.append(ind.end)
            gap = d.add(elm.Line().at(ind.end).right(0.55))
            here = (gap.end.x, gap.end.y)
        crx = d.add(elm.Line().at(here).right(1.5))
        d.add(elm.Capacitor().down().at(crx.end))
        d.add(elm.Ground())
        note(d, crx.end.x + 0.75, crx.end.y - 1.15, 'CRX')
        d.add(elm.Dot().at(crx.end))
        note(d, crx.end.x, crx.end.y + 0.55, f'rx{tag}')

        rx = cell(d, 'RX', [ipin('ain', 'left', 1.2)],
                  at=(crx.end.x + 2.4, tx.tx.y), anchor='ain', size=(2.4, 2.0))
        wire(d, crx.end, rx.ain)
        return nodes

    n0 = lane('0', 16.6, 14.4)
    n1 = lane('1', 7.6, 5.4)
    for a, b in zip(n0, n1):
        d.add(elm.Capacitor().endpoints(a, b))
        note(d, a.x + 0.55, (a.y + b.y) / 2, 'Cc/3')

    note(d, 0.3, 1.55, 'TX cell = 2 CMOS predrivers + 8 header/footer cells.  Channel is 3-section RLC + CRX.')
    note(d, 0.3, 0.90, 'RX is behavioral Python, not MOS.')
    save(d, path)


if __name__ == '__main__':
    draw_adc(ROOT / 'schematics_adc.jpg')
    draw_d2d(ROOT / 'schematics_d2d.jpg')
    draw_adc_cells(ROOT / 'schematics_adc_cells.jpg')
    draw_d2d_cells(ROOT / 'schematics_d2d_cells.jpg')
