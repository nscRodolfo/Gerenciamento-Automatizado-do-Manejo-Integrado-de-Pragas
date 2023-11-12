package my.aplication.manejointeligentedepragas.model;

public class PropriedadeModel {
    private int Cod_Propriedade;
    private String Nome;
    private String Cidade;
    private String Estado;
    private int fk_Cod_Produtor;

    public PropriedadeModel(int cod_Propriedade, String nome, String cidade, String estado, int fk_Cod_Produtor) {
        this.Cod_Propriedade = cod_Propriedade;
        this.Nome = nome;
        this.Cidade = cidade;
        this.Estado = estado;
        this.fk_Cod_Produtor = fk_Cod_Produtor;
    }

    public PropriedadeModel() {
        this.Cod_Propriedade = 0;
        this.Nome = null;
        this.Cidade = null;
        this.Estado = null;
        this.fk_Cod_Produtor = 0;
    }

    public int getCod_Propriedade() {
        return Cod_Propriedade;
    }

    public void setCod_Propriedade(int cod_Propriedade) {
        this.Cod_Propriedade = cod_Propriedade;
    }

    public String getNome() {
        return Nome;
    }

    public void setNome(String nome) {
        this.Nome = nome;
    }

    public String getCidade() {
        return Cidade;
    }

    public void setCidade(String cidade) {
        this.Cidade = cidade;
    }

    public String getEstado() {
        return Estado;
    }

    public void setEstado(String estado) {
        this.Estado = estado;
    }

    public int getFk_Cod_Produtor() {
        return fk_Cod_Produtor;
    }

    public void setFk_Cod_Produtor(int fk_Cod_Produtor) {
        this.fk_Cod_Produtor = fk_Cod_Produtor;
    }
}
