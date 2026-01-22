import argparse
from readme_renderer.markdown import render as render_md
from readme_renderer.rst import render as render_rst
from readme_renderer.txt import render as render_txt
import pathlib
from importlib.metadata import metadata
import sys
from typing import Optional, List


def main(cli_args: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Renders a .md, .rst, or .txt README to HTML",
    )
    parser.add_argument("-p", "--package", help="Get README from package metadata",
                        action="store_true")
    parser.add_argument("-f", "--format", choices=["md", "rst", "txt"],
                        help="README format (inferred from input file name or package)")
    parser.add_argument('input', help="Input README file or package name")
    parser.add_argument('-o', '--output', help="Output file (default: stdout)",
                        default='-')
    args = parser.parse_args(cli_args)

    content_format = args.format
    if args.package:
        message = metadata(args.input)
        source = message.get_payload()  # type: ignore[attr-defined] # noqa: E501 https://peps.python.org/pep-0566/

        # Infer the format of the description from package metadata.
        if not content_format:
            content_type = message.get("Description-Content-Type", "text/x-rst")
            if content_type == "text/x-rst":
                content_format = "rst"
            elif content_type == "text/markdown":
                content_format = "md"
            elif content_type == "text/plain":
                content_format = "txt"
            else:
                raise ValueError(f"invalid content type {content_type} for package "
                                 "`long_description`")
    else:
        filename = pathlib.Path(args.input)
        content_format = content_format or filename.suffix.lstrip(".")
        with filename.open() as fp:
            source = fp.read()

    if content_format == "md":
        rendered = render_md(source, stream=sys.stderr)
    elif content_format == "rst":
        rendered = render_rst(source, stream=sys.stderr)
    elif content_format == "txt":
        rendered = render_txt(source, stream=sys.stderr)
    else:
        raise ValueError(f"invalid README format: {content_format} (expected `md`, "
                         "`rst`, or `txt`)")
    if rendered is None:
        sys.exit(1)
    if args.output == "-":
        print(rendered, file=sys.stdout)
    else:
        with open(args.output, "w") as fp:
            print(rendered, file=fp)


if __name__ == '__main__':
    main()
