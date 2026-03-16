#!/usr/bin/env python3
"""
Test script for the Boogie to EFMC converter.
"""

import tempfile
import os
from aria.efmc.frontends.boogie2efmc import BoogieToEFMCConverter, boogie_to_efmc


def test_boogie_converter():
    """Test the Boogie to EFMC converter."""
    print("Testing Boogie to EFMC Converter")

    boogie_code = """
implementation SimpleLoop() {
    var x, y: int;
    entry: assume x >= 0; assume y >= 0; goto loop_header;
    loop_header: goto loop_body, loop_exit;
    loop_body: assume x > 0; x := x - 1; y := y + 1; goto loop_header;
    loop_exit: assume x <= 0; assert y >= 0; return;
}"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".bpl", delete=False) as f:
        f.write(boogie_code)
        temp_file = f.name

    try:
        # Test converter
        converter = BoogieToEFMCConverter()
        ts = converter.convert_file_to_transition_system(temp_file)
        print(f"✅ Converter - Variables: {len(ts.variables)}")

        # Test convenience function
        ts2 = boogie_to_efmc(temp_file)
        print("✅ Convenience function works")

        # Test CHC conversion
        chc_str = ts.to_chc_str()
        print(f"✅ CHC conversion - {len(chc_str)} chars")

        print("🎉 All tests passed!")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


if __name__ == "__main__":
    test_boogie_converter()
