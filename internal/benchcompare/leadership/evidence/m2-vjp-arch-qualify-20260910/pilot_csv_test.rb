require 'minitest/autorun'
require 'tmpdir'
require_relative 'build_pilot_csv'

class PilotCSVTest < Minitest::Test
  def setup
    @text = File.binread(File.join(__dir__, 'pilot.csv'))
    @rows = CSV.parse(@text)
  end

  def encode(rows)
    CSV.generate { |csv| rows.each { |row| csv << row } }
  end

  def test_complete_fixture_and_retained_export
    rows = PilotCSV.validate(@text)
    assert_equal 224, rows.length
    %w[old new].each do |arm|
      output = PilotCSV.benchstat(rows, arm)
      assert_equal 56, output.lines.length
      assert output.lines.all? { |line| line.start_with?('BenchmarkUnaryVJPBounds') }
    end
    assert_raises(ArgumentError) { PilotCSV.benchstat(rows, 'unknown') }
  end

  def test_rejects_wrong_schema_width_count_and_order
    mutations = [
      ->(r) { r[0][0] = 'private_path' },
      ->(r) { r[1] << 'extra' },
      ->(r) { r[1].pop },
      ->(r) { r.pop },
      ->(r) { r << r.last.dup },
      ->(r) { r[1], r[2] = r[2], r[1] },
      ->(r) { r[3] = r[1].dup }
    ]
    mutations.each do |mutate|
      rows = @rows.map(&:dup)
      mutate.call(rows)
      assert_raises(ArgumentError) { PilotCSV.validate(encode(rows)) }
    end
    assert_raises(ArgumentError) { PilotCSV.validate('x' * (32 * 1024 + 1)) }
  end

  def test_rejects_each_wrong_identity_and_invalid_metric
    [0, 1, 2, 3, 4].each do |column|
      rows = @rows.map(&:dup)
      rows[1][column] = 'unexpected'
      assert_raises(ArgumentError) { PilotCSV.validate(encode(rows)) }
    end
    [nil, '0', '-1', '01', '1.0', 'NaN', '1e6', '12 secret'].each do |bad|
      (5..8).each do |column|
        rows = @rows.map(&:dup)
        rows[1][column] = bad
        assert_raises(ArgumentError) { PilotCSV.validate(encode(rows)) }
      end
    end
    [7, 8].each do |column|
      rows = @rows.map(&:dup)
      rows[1][column] = '123'
      assert_raises(ArgumentError) { PilotCSV.validate(encode(rows)) }
    end
  end

  def with_captures
    Dir.mktmpdir('pilot-sanitizer-test-') do |directory|
      @rows.drop(1).each_slice(16).with_index do |rows, invocation|
        stem = File.join(directory, format('%03d', invocation))
        lines = rows.map do |row|
          kind, _relu, dtype, size = row[4].split('_')
          name = "BenchmarkUnaryVJPBounds#{kind == 'taped' ? 'Tape' : ''}/ReLU/#{dtype}/#{size}"
          "#{name} #{row[5]} #{row[6]} ns/op 1.0 MB/s #{row[7]} B/op #{row[8]} allocs/op\n"
        end
        stdout = "cpu: Synthetic test host\n" + lines.join + "PASS\n"
        metadata = {'pair' => invocation / 2 + 1, 'arm' => PilotCSV.arm(invocation),
                    'campaign' => 1, 'build' => 'default', 'procs' => 1, 'scope' => 'autograd',
                    'exit_code' => 0, 'signal' => nil,
                    'stdout_sha256' => Digest::SHA256.hexdigest(stdout),
                    'stderr_sha256' => Digest::SHA256.hexdigest(''),
                    'env' => {'PRIVATE_TEST_SENTINEL' => 'do-not-export'},
                    'argv' => ['private-compiler-path']}
        File.binwrite("#{stem}.stdout", stdout)
        File.binwrite("#{stem}.stderr", '')
        File.write("#{stem}.json", JSON.generate(metadata))
      end
      yield directory
    end
  end

  def test_sanitizer_round_trip_omits_metadata
    with_captures do |directory|
      result = PilotSanitizer.build(directory)
      assert_equal @text, result
      refute_includes result, 'do-not-export'
      refute_includes result, 'private-compiler-path'
    end
  end

  def test_sanitizer_rejects_missing_extra_corrupt_and_failed_captures
    mutations = [
      ->(dir) { File.rename(File.join(dir, '000.stdout'), File.join(dir, 'missing.stdout')) },
      ->(dir) { File.write(File.join(dir, '014.json'), '{}') },
      ->(dir) { File.write(File.join(dir, '000.stdout'), 'corrupt') },
      ->(dir) { File.write(File.join(dir, '000.stderr'), 'failed') },
      ->(dir) do
        file = File.join(dir, '000.json')
        metadata = JSON.parse(File.read(file))
        metadata['exit_code'] = 1
        File.write(file, JSON.generate(metadata))
      end,
      ->(dir) do
        file = File.join(dir, '000.json')
        metadata = JSON.parse(File.read(file))
        metadata['arm'] = 'new'
        File.write(file, JSON.generate(metadata))
      end
    ]
    mutations.each do |mutate|
      with_captures do |directory|
        mutate.call(directory)
        assert_raises(ArgumentError) { PilotSanitizer.build(directory) }
      end
    end
  end
end
