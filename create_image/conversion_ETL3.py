def main(in_file, out_file):
    masks = list(b'\x00\x30\x3c\x3f')

    try:
        for num in range(73):  #72*3=216byte
            data = [0]
            data.extend(in_file.read(3))
            data.append(0)

            for lhs, mask, i, rhs in zip(data, masks, range(0, 8, 2), data[1:]):
                # ((lhs << 6 - (i * 2 )) & mask | rhs >> (i * 2 + 2)
                d = ((lhs<<(6 - i)) & mask) | (rhs>>(i + 2))

                out_file.write(d.to_bytes(1, 'little'))

        read_data = in_file.read(2736)
        out_file.write(read_data)
    except IOError:
        print("cannot read data")
        raise

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python {} <in_file> <out_file>".format(sys.argv[0]))
        sys.exit(1)
        
    with open(sys.argv[1], 'rb') as f_in:
        with open(sys.argv[2], 'wb+') as f_out:
            main(f_in, f_out)
