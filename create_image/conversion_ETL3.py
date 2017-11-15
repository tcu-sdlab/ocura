ETL3_FILESIZE = 2736 + 216


def convert_sample(in_stream, out_stream):
    masks = list(b'\x00\x30\x3c\x3f')

    try:
        for _ in range(73):  #72*3=216byte
            data = [0]
            data.extend(in_stream.read(3))
            data.append(0)

            for lhs, mask, i, rhs in zip(data, masks, range(0, 8, 2), data[1:]):
                # ((lhs << 6 - (i * 2 )) & mask | rhs >> (i * 2 + 2)
                d = ((lhs<<(6 - i)) & mask) | (rhs>>(i + 2))

                out_stream.write(d.to_bytes(1, 'little'))

        read_data = in_stream.read(2736)
        out_stream.write(read_data)
    except IOError:
        print("cannot read data")
        raise


def main(in_file, in_size, out_file):
    for _ in range(in_size / ETL3_FILESIZE):
        convert_sample(in_file, out_file)


if __name__ == '__main__':
    import sys
    import os
    if len(sys.argv) != 3:
        print("Usage: python {} <in_file> <out_file>".format(sys.argv[0]))
        sys.exit(1)

    f_in_size = os.stat(sys.argv[1]).st_size 
    if f_in_size != ETL3_FILESIZE:
        error_msg = "{} has invalid file size (except {} bytes)"
        print(error_msg.format(sys.argv[1], ETL3_FILESIZE))
        sys.exit(1)

    with open(sys.argv[1], 'rb') as f_in:
        with open(sys.argv[2], 'wb+') as f_out:
            main(f_in, f_in_size, f_out)
