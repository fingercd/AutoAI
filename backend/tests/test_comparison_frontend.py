from pathlib import Path
import subprocess


def test_comparison_page_pure_functions():
    result=subprocess.run(['node','--test',str(Path(__file__).with_name('frontend_comparison_page.mjs'))],capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0,result.stdout+result.stderr
