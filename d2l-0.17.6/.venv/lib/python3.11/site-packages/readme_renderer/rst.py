# Copyright 2014 Donald Stufft
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
from typing import Any, Dict, IO, Optional, Union

from docutils.core import publish_parts
from docutils.nodes import colspec, image
from docutils.writers.html5_polyglot import HTMLTranslator, Writer
from docutils.utils import SystemMessage

from .clean import clean


class ReadMeHTMLTranslator(HTMLTranslator):  # type: ignore[misc] # docutils is incomplete, returns `Any` python/typeshed#7256 # noqa E501

    # Overrides base class not to output `<object>` tag for SVG images.
    object_image_types: Dict[str, str] = {}

    def emptytag(
        self,
        node: Union[colspec, image],
        tagname: str,
        suffix: str = "\n",
        **attributes: Any
    ) -> Any:
        """Override this to add back the width/height attributes."""
        if tagname == "img":
            if "width" in node:
                attributes["width"] = node["width"]
            if "height" in node:
                attributes["height"] = node["height"]

        return super().emptytag(
            node, tagname, suffix, **attributes
        )


SETTINGS = {
    # Cloaking email addresses provides a small amount of additional
    # privacy protection for email addresses inside of a chunk of ReST.
    "cloak_email_addresses": True,

    # Prevent a lone top level heading from being promoted to document
    # title, and thus second level headings from being promoted to top
    # level.
    "doctitle_xform": True,

    # Prevent a lone subsection heading from being promoted to section
    # title, and thus second level headings from being promoted to top
    # level.
    "sectsubtitle_xform": True,

    # Set our initial header level
    "initial_header_level": 2,

    # Prevent local files from being included into the rendered output.
    # This is a security concern because people can insert files
    # that are part of the system, such as /etc/passwd.
    "file_insertion_enabled": False,

    # Halt rendering and throw an exception if there was any errors or
    # warnings from docutils.
    "halt_level": 2,

    # Output math blocks as LaTeX that can be interpreted by MathJax for
    # a prettier display of Math formulas.
    # Pass a dummy path to supress docutils warning and emit HTML.
    "math_output": "MathJax /dummy.js",

    # Disable raw html as enabling it is a security risk, we do not want
    # people to be able to include any old HTML in the final output.
    "raw_enabled": False,

    # Disable all system messages from being reported.
    "report_level": 5,

    # Use typographic quotes, and transform --, ---, and ... into their
    # typographic counterparts.
    "smart_quotes": True,

    # Strip all comments from the rendered output.
    "strip_comments": True,

    # Use the short form of syntax highlighting so that the generated
    # Pygments CSS can be used to style the output.
    "syntax_highlight": "short",

    # Maximum width (in characters) for one-column field names.
    # 0 means "no limit"
    "field_name_limit": 0,
}


def render(
    raw: str,
    stream: Optional[IO[str]] = None,
    **kwargs: Any
) -> Optional[str]:
    if stream is None:
        # Use a io.StringIO as the warning stream to prevent warnings from
        # being printed to sys.stderr.
        stream = io.StringIO()

    settings = SETTINGS.copy()
    settings["warning_stream"] = stream

    writer = Writer()
    writer.translator_class = ReadMeHTMLTranslator

    try:
        parts = publish_parts(raw, writer=writer, settings_overrides=settings)
    except SystemMessage:
        rendered = None
    else:
        rendered = parts.get("docinfo", "") + parts.get("fragment", "")

    if rendered:
        return clean(rendered)
    else:
        # If the warnings stream is empty, docutils had none, so add ours.
        if not stream.tell():
            stream.write("No content rendered from RST source.")
        return None
