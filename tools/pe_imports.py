"""Lists the import (and delay-import) DLL names of PE files. Used by tools/check_deps.py."""
import struct, sys
def imports(data):
    pe = struct.unpack_from('<I', data, 0x3c)[0]
    assert data[pe:pe+4] == b'PE\0\0'
    nsec = struct.unpack_from('<H', data, pe+6)[0]
    optsz = struct.unpack_from('<H', data, pe+20)[0]
    opt = pe+24
    magic = struct.unpack_from('<H', data, opt)[0]
    dd_off = opt + (112 if magic == 0x20b else 96)
    secs = []
    so = opt + optsz
    for i in range(nsec):
        s = so + i*40
        name, vsz, va, rsz, rp = struct.unpack_from('<8sIIII', data, s)
        secs.append((va, vsz, rp, rsz))
    def rva2off(rva):
        for va, vsz, rp, rsz in secs:
            if va <= rva < va + max(vsz, rsz): return rva - va + rp
        return None
    out = []
    for idx, name in ((1, 'import'), (13, 'delay')):
        rva, size = struct.unpack_from('<II', data, dd_off + idx*8)
        if not rva: continue
        off = rva2off(rva); i = 0
        while True:
            if idx == 1:
                ent = struct.unpack_from('<IIIII', data, off + i*20); name_rva = ent[3]
            else:
                ent = struct.unpack_from('<IIIIIIII', data, off + i*32); name_rva = ent[1]
            if name_rva == 0: break
            o = rva2off(name_rva); e = data.index(b'\0', o)
            out.append((name, data[o:e].decode()))
            i += 1
    return out
if __name__ == "__main__":
  for f in sys.argv[1:]:
    d = open(f, 'rb').read()
    print("==", f)
    for kind, n in imports(d): print("  ", kind, n)
