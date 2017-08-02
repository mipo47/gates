from gates.dataset import DataSet

def read_proben1(filename):
    train_data = DataSet()
    valid_data = DataSet()
    test_data = DataSet()

    with open(filename) as infile:
        lines = infile.readlines()
        bool_in = int(lines[0].split('=')[-1])
        real_in = int(lines[1].split('=')[-1])
        total_in = bool_in + real_in
        training_examples = int(lines[4].split('=')[-1])
        validation_examples = int(lines[5].split('=')[-1])
        test_examples = int(lines[6].split('=')[-1])

        for line in lines[7:]: # skip header
            line = line.split()

            # Records = offset (x0) + remaining data points
            input = [float(x) for x in line[:total_in]]
            output = [float(x) for x in line[total_in:]]

            # Determine what data set to put this in
            if train_data.length() < training_examples:
                train_data.add(input, output)
            elif valid_data.length() < validation_examples:
                valid_data.add(input, output)
            elif test_data.length() < test_examples:
                test_data.add(input, output)
            else:
                break

    train_data.to_numpy()
    valid_data.to_numpy()
    test_data.to_numpy()
    return train_data, valid_data, test_data