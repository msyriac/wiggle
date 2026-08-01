=========
Releasing
=========

``wiggle`` publishes to PyPI as ``pywiggle``. Releases are automated: the
``upload_pypi`` job in ``.github/workflows/build.yml`` fires on any pushed tag
starting with ``v`` and uploads the wheels and source distribution built earlier
in the same run.

Release checklist
=================

1. Bump the version
-------------------

Edit ``pyproject.toml``:

.. code-block:: toml

   [project]
   name = "pywiggle"
   version = "0.1.19"

Use a number that has never been uploaded. PyPI filenames are permanent: once
``pywiggle-0.1.19`` exists, that version is burned forever, even if the release
is later deleted.

2. Check the metadata locally
-----------------------------

.. code-block:: console

   $ pip install --upgrade build twine
   $ rm -rf dist
   $ python -m build --sdist
   $ twine check --strict dist/*

``twine check`` must print ``PASSED``. If it complains about the README or the
license metadata, fix ``pyproject.toml`` and repeat. Do not tag until it is
clean.

3. Commit the bump
------------------

.. code-block:: console

   $ git add pyproject.toml
   $ git commit -m "Release 0.1.19"
   $ git push origin main

4. Tag and push
---------------

Read the version out of ``pyproject.toml`` rather than retyping it, so the tag
cannot drift from the metadata:

.. code-block:: console

   $ VERSION=$(grep -m1 '^version' pyproject.toml | cut -d'"' -f2)
   $ echo "Tagging v$VERSION"
   $ git tag -a "v$VERSION" -m "pywiggle $VERSION"
   $ git push origin "v$VERSION"

5. Watch the run
----------------

The tag push starts the release workflow. Wheels are built for Linux and macOS,
the sdist on Linux, and ``upload_pypi`` publishes all of them together. The
``test-wheel`` job then installs the freshly published version from PyPI and
runs the test suite against it, polling for up to five minutes while the CDN
catches up.

