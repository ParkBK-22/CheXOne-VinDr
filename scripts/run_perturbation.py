import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", type=str, required=True)
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["lesion", "non_lesion"], required=True)
    args = parser.parse_args()

    raise NotImplementedError(
        "Perturbation is not implemented yet. "
        "This script will later create lesion and non-lesion perturbed images."
    )


if __name__ == "__main__":
    main()
