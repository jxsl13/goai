require "minitest/autorun"
require "tmpdir"
require "stringio"
require_relative "run"

class UnaryBCECampaignTest < Minitest::Test
  C = UnaryBCECampaign

  def fixture(expected, procs: 1, scope: "autograd")
    suffix = procs == 1 ? "" : "-#{procs}"
    head = "goos: darwin\ngoarch: arm64\npkg: github.com/jxsl13/goai/#{scope}\ncpu: Apple M2 Pro\n"
    rows = expected.flat_map do |name|
      ["#{name}#{suffix} 10 20 ns/op 10.0 MB/s 32 B/op 2 allocs/op",
       "#{name}#{suffix} 20 10 ns/op 20.0 MB/s 32 B/op 2 allocs/op"]
    end
    head + rows.join("\n") + "\nPASS\n"
  end

  def test_full_and_pilot_cell_sets
    assert_equal 8, C.cells("autograd", true).length
    assert_equal 48, C.cells("autograd", false).length
    assert_equal 3, C.cells("nn", false).length
    assert_equal C.cells("autograd", false).uniq, C.cells("autograd", false)
    assert_includes C.cells("autograd", false), "BenchmarkUnaryVJPBounds/Log/F64/N0"
    refute_includes C.cells("autograd", false), "BenchmarkUnaryVJPBoundsTape/Log/F64/N0"
  end

  def test_warmup_removed_per_cell_not_per_process
    expected = C.cells("autograd", true)
    rows = C.parse(fixture(expected), expected, 1, "autograd")
    assert_equal expected.length, rows.length
    assert rows.all? { |row| row.include?(" 20 10 ns/op ") }
  end

  def test_procs12_and_training_rows
    expected = C.cells("nn", false)
    rows = C.parse(fixture(expected, procs: 12, scope: "nn"), expected, 12, "nn")
    assert_equal 3, rows.length
    assert rows.all? { |row| row.include?("-12 ") }
  end

  def test_mixed_warmup_order_remains_per_cell
    expected = C.cells("autograd", true)
    lines = fixture(expected).lines
    rows = lines.grep(/^Benchmark/)
    shuffled = lines.reject { |line| line.start_with?("Benchmark", "PASS") } +
      rows.each_slice(2).map(&:first) + rows.each_slice(2).map(&:last) + ["PASS\n"]
    assert C.parse(shuffled.join, expected, 1, "autograd").all? { |row| row.include?(" 20 10 ns/op ") }
  end

  def test_missing_extra_or_duplicate_sample_is_invalid
    expected = C.cells("autograd", true)
    text = fixture(expected)
    row = text.lines.find { |line| line.start_with?("Benchmark") }
    [text.sub(row, ""), text.sub("PASS", row + "PASS"),
     text.sub("PASS", row.sub(expected.first, "BenchmarkUnexpected") + "PASS")].each do |bad|
      assert_raises(C::Invalid) { C.parse(bad, expected, 1, "autograd") }
    end
    all_but_one = text.lines.reject { |line| line.start_with?(expected.first + " ") }.join
    assert_raises(C::Invalid) { C.parse(all_but_one, expected, 1, "autograd") }
  end

  def test_empty_success_is_not_a_measurement
    expected = C.cells("autograd", true)
    assert_raises(C::Invalid) { C.parse("PASS\n", expected, 1, "autograd") }
  end

  def test_bad_pass_or_host_headers_are_invalid
    expected = C.cells("autograd", true)
    text = fixture(expected)
    [text.sub("PASS\n", ""), text + "PASS\n", text + "FAIL\n",
     text + "--- FAIL: synthetic\n", text + "panic: synthetic\n",
     text.sub("arm64", "amd64"), text.sub("Apple M2 Pro", "other"),
     text.sub("goos: darwin\n", ""), text + "cpu: Apple M2 Pro\n",
     text + "goos: linux\n", text + "goarch: amd64\n",
     text + "cpu: Other CPU\n", text + "pkg: example.invalid/other\n"].each do |bad|
      assert_raises(C::Invalid) { C.parse(bad, expected, 1, "autograd") }
    end
    assert_raises(C::Invalid) { C.parse(text, expected, 12, "autograd") }
  end

  def test_nonnumeric_nonfinite_and_negative_metrics_are_invalid
    expected = C.cells("autograd", true)
    %w[NaN Infinity -Infinity 1e999 -1e999 bogus -1].each do |value|
      text = fixture(expected).sub("20 ns/op", "#{value} ns/op")
      assert_raises(C::Invalid) { C.parse(text, expected, 1, "autograd") }
    end
  end

  def test_missing_fractional_or_duplicate_metrics_are_invalid
    expected = C.cells("autograd", true)
    text = fixture(expected)
    [text.sub("20 ns/op", "0 ns/op"), text.sub("32 B/op", ""),
     text.sub("2 allocs/op", ""), text.sub("32 B/op", "3.5 B/op"),
     text.sub("2 allocs/op", "1.5 allocs/op"),
     text.sub("32 B/op", "32 B/op 32 B/op"),
     text.sub("20 ns/op", "20 ns/op orphan"),
     text.sub(" 10 20 ns/op", " 0 20 ns/op")].each do |bad|
      assert_raises(C::Invalid) { C.parse(bad, expected, 1, "autograd") }
    end
  end

  def test_plan_is_paired_complete_and_alternating
    assert_equal 14, C.plan(true).length
    plan = C.plan(false)
    assert_equal 336, plan.length
    assert_equal 336, plan.uniq.length
    plan.each_slice(2) do |first, second|
      assert_equal first.reject { |key, _| key == :arm }, second.reject { |key, _| key == :arm }
      assert_equal %w[new old], [first[:arm], second[:arm]].sort
    end
    groups = plan.group_by { |entry| entry.reject { |key, _| key == :pair || key == :arm } }
    assert_equal 24, groups.length
    groups.each_value do |entries|
      %w[old new].each do |arm|
        assert_equal (1..7).to_a, entries.select { |entry| entry[:arm] == arm }.map { |entry| entry[:pair] }
      end
      first_arms = entries.each_slice(2).map { |pair| pair.first[:arm] }
      assert first_arms.each_cons(2).all? { |a, b| a != b }
    end
    assert_equal %w[old new], plan.select { |e| e[:campaign] == 1 && e[:pair] == 1 }.first(2).map { |e| e[:arm] }
    assert_equal %w[new old], plan.select { |e| e[:campaign] == 2 && e[:pair] == 1 }.first(2).map { |e| e[:arm] }
    assert_equal ["simd", 12], plan.find { |e| e[:campaign] == 2 }.values_at(:build, :procs)
  end

  def fake_roots(base, code)
    %w[old new].map do |arm|
      root = File.join(base, arm)
      Dir.mkdir(root)
      binary = File.join(root, "default-autograd.test")
      File.write(binary, "#!/usr/bin/env ruby\n" + code + "\n")
      File.chmod(0755, binary)
      root
    end
  end

  def test_fake_process_campaign_keeps_all_raw_samples
    Dir.mktmpdir("unary-bce-runner-test-") do |base|
      expected = C.cells("autograd", true)
      argv = ["-test.run", "^$", "-test.bench", C.pattern("autograd", true),
              "-test.benchmem", "-test.benchtime=200ms", "-test.count=2"]
      code = "abort 'arguments' unless ARGV == #{argv.inspect}\n" +
             "abort 'procs' unless ENV['GOMAXPROCS'] == '1'\n" +
             "print #{fixture(expected).dump}"
      roots = fake_roots(base, code)
      out = File.join(base, "evidence")
      previous_stdout = $stdout
      begin
        $stdout = StringIO.new
        C.run(*roots, out, "pilot")
      ensure
        $stdout = previous_stdout
      end
      manifest = JSON.parse(File.read(File.join(out, "manifest.json")))
      assert_equal "complete", manifest.fetch("status")
      assert_equal C::CHILD_ENV, manifest.fetch("child_environment")
      assert_equal true, manifest.fetch("unsetenv_others")
      assert_includes manifest.fetch("qualification"), "not evaluated"
      assert_equal 14, Dir[File.join(out, "*.stdout")].length
      assert_equal 14, Dir[File.join(out, "*.stderr")].length
      assert_equal 15, Dir[File.join(out, "*.json")].length
      assert_equal 224, Dir[File.join(out, "*.stdout")].sum { |path| File.readlines(path).grep(/^Benchmark/).length }
      retained = File.readlines(File.join(out, "retained.txt")).grep(/^Benchmark/)
      assert_equal 112, retained.length
      assert retained.all? { |row| row.include?(" 20 10 ns/op ") }
      before = Digest::SHA256.file(File.join(out, "manifest.json")).hexdigest
      assert_raises(Errno::EEXIST) { C.run(*roots, out, "pilot") }
      assert_equal before, Digest::SHA256.file(File.join(out, "manifest.json")).hexdigest
    end
  end

  def test_failed_process_retains_exit_and_raw_output
    Dir.mktmpdir("unary-bce-runner-fail-") do |base|
      roots = fake_roots(base, "puts 'partial output'; warn 'synthetic error'; exit 9")
      out = File.join(base, "evidence")
      assert_raises(C::Invalid) { C.run(*roots, out, "pilot") }
      assert_equal "failed", JSON.parse(File.read(File.join(out, "manifest.json"))).fetch("status")
      invocation = JSON.parse(File.read(File.join(out, "000.json")))
      assert_equal 9, invocation.fetch("exit_code")
      assert_equal "partial output\n", File.read(File.join(out, "000.stdout"))
      assert_equal "synthetic error\n", File.read(File.join(out, "000.stderr"))
      refute File.exist?(File.join(out, "retained.txt"))
    end
  end

  def quietly
    previous = $stdout
    $stdout = StringIO.new
    yield
  ensure
    $stdout = previous
  end

  def test_complete_child_environment_is_recorded_without_inheriting_secrets
    Dir.mktmpdir("unary-bce-environment-") do |base|
      child_env = C::CHILD_ENV.merge("GOMAXPROCS" => "1")
      code = "abort 'unexpected environment' unless ENV.to_h == #{child_env.inspect}\n" +
             "print #{fixture(C.cells('autograd', true)).dump}"
      roots = fake_roots(base, code)
      out = File.join(base, "evidence")
      sentinel = "UNARY_BCE_TEST_PRIVATE_SENTINEL"
      previous = ENV[sentinel]
      ENV[sentinel] = "must-not-reach-child-or-evidence"
      begin
        quietly { C.run(*roots, out, "pilot") }
      ensure
        previous.nil? ? ENV.delete(sentinel) : ENV[sentinel] = previous
      end
      Dir[File.join(out, "*.json")].reject { |p| p.end_with?("manifest.json") }.each do |path|
        invocation = JSON.parse(File.read(path))
        assert_equal child_env, invocation.fetch("env")
        assert_equal true, invocation.fetch("unsetenv_others")
        refute_includes File.read(path), sentinel
        assert_equal invocation.fetch("binary_sha256_before"), invocation.fetch("binary_sha256_after")
      end
      refute_includes File.read(File.join(out, "manifest.json")), sentinel
    end
  end

  def test_final_invocation_binary_mutation_and_removal_retain_raw_failure
    ["File.open(__FILE__, 'a') { |f| f.puts '# changed' }", "File.unlink(__FILE__)"].each do |mutation|
      Dir.mktmpdir("unary-bce-final-change-") do |base|
        code = <<~RUBY
          state = __FILE__ + '.count'
          count = File.exist?(state) ? File.read(state).to_i + 1 : 1
          File.write(state, count)
          #{mutation} if count == 7 && File.basename(File.dirname(__FILE__)) == 'new'
          print #{fixture(C.cells('autograd', true)).dump}
        RUBY
        roots = fake_roots(base, code)
        out = File.join(base, "evidence")
        error = assert_raises(C::Invalid) { quietly { C.run(*roots, out, "pilot") } }
        assert_match(/binary changed after invocation 13/, error.message)
        assert_equal "failed", JSON.parse(File.read(File.join(out, "manifest.json"))).fetch("status")
        record = JSON.parse(File.read(File.join(out, "013.json")))
        assert_equal 0, record.fetch("exit_code")
        refute_equal record.fetch("binary_sha256_before"), record.fetch("binary_sha256_after")
        assert_equal Digest::SHA256.file(File.join(out, "013.stdout")).hexdigest, record.fetch("stdout_sha256")
        assert_equal 104, File.readlines(File.join(out, "retained.txt")).grep(/^Benchmark/).length
      end
    end
  end

  def test_final_sweep_detects_change_to_previously_measured_binary
    Dir.mktmpdir("unary-bce-final-sweep-") do |base|
      code = <<~RUBY
        state = __FILE__ + '.count'
        count = File.exist?(state) ? File.read(state).to_i + 1 : 1
        File.write(state, count)
        if count == 7 && File.basename(File.dirname(__FILE__)) == 'new'
          other = File.join(File.dirname(File.dirname(__FILE__)), 'old', 'default-autograd.test')
          File.open(other, 'a') { |f| f.puts '# changed after its last use' }
        end
        print #{fixture(C.cells('autograd', true)).dump}
      RUBY
      roots = fake_roots(base, code)
      out = File.join(base, "evidence")
      error = assert_raises(C::Invalid) { quietly { C.run(*roots, out, "pilot") } }
      assert_match(/binary changed before completion/, error.message)
      assert_equal "failed", JSON.parse(File.read(File.join(out, "manifest.json"))).fetch("status")
      assert_equal 14, Dir[File.join(out, "*.stdout")].length
    end
  end

  def test_successful_process_with_missing_cells_is_failed_campaign
    Dir.mktmpdir("unary-bce-runner-empty-") do |base|
      roots = fake_roots(base, "puts 'PASS'")
      out = File.join(base, "evidence")
      assert_raises(C::Invalid) { C.run(*roots, out, "pilot") }
      assert_equal "failed", JSON.parse(File.read(File.join(out, "manifest.json"))).fetch("status")
      assert_equal 0, JSON.parse(File.read(File.join(out, "000.json"))).fetch("exit_code")
      refute File.exist?(File.join(out, "retained.txt"))
    end
  end
end
